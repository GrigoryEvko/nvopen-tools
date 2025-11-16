// Function: sub_1BF96D0
// Address: 0x1bf96d0
//
__int64 __fastcall sub_1BF96D0(__int64 a1, unsigned int a2, int a3)
{
  unsigned int *v3; // rax
  unsigned int v4; // r14d
  __int64 v5; // r8
  int v6; // r9d
  __int64 v7; // rcx
  unsigned int v8; // edx
  __int64 v9; // rcx
  __int64 v10; // r8
  int v11; // r9d
  int *v12; // r8
  int v13; // ebx
  int v14; // eax
  unsigned int v15; // r12d
  int *v16; // rsi
  __int64 v17; // rdi
  unsigned int v18; // ecx
  __int64 v19; // rax
  unsigned int v22; // [rsp+14h] [rbp-8Ch] BYREF
  unsigned __int64 v23; // [rsp+18h] [rbp-88h] BYREF
  int *v24; // [rsp+20h] [rbp-80h] BYREF
  __int64 v25; // [rsp+28h] [rbp-78h]
  _BYTE v26[112]; // [rsp+30h] [rbp-70h] BYREF

  v3 = (unsigned int *)sub_16D40F0((__int64)qword_4FBB430);
  if ( v3 )
    v4 = *v3;
  else
    v4 = qword_4FBB430[2];
  v23 = sub_1BF9660(a1);
  if ( (_DWORD)v23 )
  {
    v7 = HIDWORD(v23);
    if ( !(a2 % 0xA) || (v8 = 16, a2 > 0x58) )
      v8 = a2 % 0xA == 0 ? 32 : 24;
    if ( HIDWORD(v23) > v8 )
      HIDWORD(v23) = 0;
    v24 = (int *)v26;
    v25 = 0x800000000LL;
  }
  else
  {
    sub_1C2EEA0(&v24, a1);
    v22 = 0;
    sub_1C2EF90(a1, &v22);
    if ( BYTE4(v24) )
    {
      v15 = (unsigned int)v24;
      if ( v22 && (unsigned int)v24 > v22 )
        v15 = v22;
    }
    else
    {
      v15 = v22;
    }
    if ( v4 )
    {
      if ( v4 > 0x100 )
        v4 = 256;
      v4 = (v4 + 7) & 0xFFFFFFF8;
    }
    v7 = (unsigned int)v23;
    v24 = (int *)v26;
    v25 = 0x800000000LL;
    if ( !(_DWORD)v23 )
    {
      if ( v15 )
        return v15;
      v15 = v4;
      if ( v4 )
        return v15;
      LODWORD(v23) = 256;
    }
  }
  if ( !sub_1BF7E40(a2, (unsigned int *)&v23, (__int64)&v24, v7, v5, v6) )
  {
    if ( !HIDWORD(v23) )
      goto LABEL_12;
    HIDWORD(v23) = 0;
    sub_1BF7E40(a2, (unsigned int *)&v23, (__int64)&v24, v9, v10, v11);
  }
  if ( !HIDWORD(v23) )
  {
LABEL_12:
    v12 = v24;
    v13 = a3 - 1;
    if ( a3 - 1 <= 0 )
      v13 = 1;
    v14 = *v24;
    v15 = *v24;
    if ( v13 > *v24 )
    {
      if ( (_DWORD)v25 )
      {
        v16 = v24 + 2;
        v17 = 0;
        while ( 1 )
        {
          v18 = v17 + 1;
          if ( v13 <= v14 || (unsigned int)v25 <= v18 )
          {
            if ( (_DWORD)v25 == v18 )
              break;
          }
          else if ( v13 <= *v16 )
          {
            v15 = v24[2 * v17];
            goto LABEL_25;
          }
          v14 = *v16;
          v17 = (int)v18;
          v16 += 2;
        }
      }
      v19 = 0x3FFFFFFFFFFFFFFELL;
      if ( v13 >= v24[2 * (unsigned int)v25 - 2] )
        v19 = 2LL * ((int)v25 - 1);
      v15 = v24[v19];
    }
    goto LABEL_25;
  }
  v12 = v24;
  v15 = v24[2 * (unsigned int)v25 - 2];
LABEL_25:
  if ( v12 != (int *)v26 )
    _libc_free((unsigned __int64)v12);
  return v15;
}
