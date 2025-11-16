// Function: sub_2C74240
// Address: 0x2c74240
//
__int64 __fastcall sub_2C74240(__int64 a1, unsigned int a2, int a3)
{
  unsigned int *v4; // rax
  unsigned int v5; // r14d
  __int64 v6; // r8
  __int64 v7; // r9
  unsigned int v8; // r15d
  __int64 v9; // rcx
  unsigned int v10; // edx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  int v14; // ebx
  _BYTE *v15; // r8
  int v16; // esi
  int *v17; // rcx
  __int64 v18; // rdi
  unsigned int v19; // eax
  __int64 v20; // rax
  __int64 v22; // [rsp+10h] [rbp-90h] BYREF
  __int64 v23; // [rsp+18h] [rbp-88h]
  _DWORD *v24; // [rsp+20h] [rbp-80h] BYREF
  __int64 v25; // [rsp+28h] [rbp-78h]
  _BYTE v26[112]; // [rsp+30h] [rbp-70h] BYREF

  v4 = (unsigned int *)sub_C94E20((__int64)qword_4F86390);
  if ( v4 )
    v5 = *v4;
  else
    v5 = qword_4F86390[2];
  v22 = sub_2C741A0(a1);
  v8 = v22;
  if ( (_DWORD)v22 )
  {
    v9 = HIDWORD(v22);
    if ( !(a2 % 0xA) || (v10 = 16, a2 > 0x58) )
      v10 = a2 % 0xA == 0 ? 32 : 24;
    if ( HIDWORD(v22) > v10 )
      HIDWORD(v22) = 0;
    v24 = v26;
    v25 = 0x800000000LL;
LABEL_10:
    if ( !sub_2C72680(a2, (unsigned int *)&v22, (__int64)&v24, v9, v6, v7) )
    {
      if ( !HIDWORD(v22) )
        goto LABEL_12;
      HIDWORD(v22) = 0;
      sub_2C72680(a2, (unsigned int *)&v22, (__int64)&v24, v11, v12, v13);
    }
    if ( HIDWORD(v22) )
    {
      v15 = v24;
      v8 = v24[2 * (unsigned int)v25 - 2];
LABEL_25:
      if ( v15 != v26 )
        _libc_free((unsigned __int64)v15);
      return v8;
    }
LABEL_12:
    v14 = a3 - 1;
    v15 = v24;
    v16 = *v24;
    if ( v14 <= 0 )
      v14 = 1;
    v8 = *v24;
    if ( v14 > v16 )
    {
      if ( (_DWORD)v25 )
      {
        v17 = v24 + 2;
        v18 = 0;
        while ( 1 )
        {
          v19 = v18 + 1;
          if ( v16 >= v14 || (unsigned int)v25 <= v19 )
          {
            if ( (_DWORD)v25 == v19 )
              break;
          }
          else if ( v14 <= *v17 )
          {
            v8 = v24[2 * v18];
            goto LABEL_25;
          }
          v16 = *v17;
          v18 = (int)v19;
          v17 += 2;
        }
      }
      v20 = 0x3FFFFFFFFFFFFFFELL;
      if ( v14 >= v24[2 * (unsigned int)v25 - 2] )
        v20 = 2LL * ((int)v25 - 1);
      v8 = v24[v20];
    }
    goto LABEL_25;
  }
  v23 = sub_CE9340(a1);
  v24 = (_DWORD *)sub_CE9180(a1);
  if ( BYTE4(v23) )
  {
    v8 = v23;
    if ( !BYTE4(v24) )
      goto LABEL_33;
  }
  else
  {
    if ( !BYTE4(v24) )
      goto LABEL_33;
    v8 = -1;
  }
  if ( v8 > (unsigned int)v24 )
    v8 = (unsigned int)v24;
LABEL_33:
  if ( v5 )
  {
    if ( v5 > 0x100 )
      v5 = 256;
    v5 = 8 * ((v5 - 1) >> 3) + 8;
  }
  v9 = (unsigned int)v22;
  v24 = v26;
  v25 = 0x800000000LL;
  if ( !v8 )
  {
    v8 = v5;
    if ( !v5 )
    {
      LODWORD(v22) = 256;
      goto LABEL_10;
    }
  }
  return v8;
}
