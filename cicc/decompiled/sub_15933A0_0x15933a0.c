// Function: sub_15933A0
// Address: 0x15933a0
//
__int64 __fastcall sub_15933A0(__int64 a1, __int64 (__fastcall *a2)(_QWORD *))
{
  _QWORD *v3; // rdx
  unsigned int v4; // eax
  _QWORD *v5; // r13
  unsigned int v6; // r15d
  __int64 v7; // rax
  _QWORD *v8; // r15
  char v9; // dl
  __int64 v10; // r14
  _QWORD *v11; // rax
  _QWORD *v12; // rsi
  _QWORD *v13; // rcx
  __int64 v14; // rax
  _QWORD *v15; // rdi
  _QWORD *v17; // [rsp+10h] [rbp-F0h] BYREF
  __int64 v18; // [rsp+18h] [rbp-E8h]
  _QWORD v19[8]; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v20; // [rsp+60h] [rbp-A0h] BYREF
  _QWORD *v21; // [rsp+68h] [rbp-98h]
  _QWORD *v22; // [rsp+70h] [rbp-90h]
  __int64 v23; // [rsp+78h] [rbp-88h]
  int v24; // [rsp+80h] [rbp-80h]
  _QWORD v25[15]; // [rsp+88h] [rbp-78h] BYREF

  v19[0] = a1;
  v18 = 0x800000001LL;
  v23 = 0x100000008LL;
  v17 = v19;
  v24 = 0;
  v20 = 1;
  v21 = v25;
  v22 = v25;
  v3 = v19;
  v25[0] = a1;
  v4 = 1;
  while ( 1 )
  {
    v5 = (_QWORD *)v3[v4 - 1];
    LODWORD(v18) = v4 - 1;
    if ( *((_BYTE *)v5 + 16) <= 3u )
    {
      v6 = a2(v5);
      if ( (_BYTE)v6 )
        break;
    }
    v7 = 24LL * (*((_DWORD *)v5 + 5) & 0xFFFFFFF);
    if ( (*((_BYTE *)v5 + 23) & 0x40) != 0 )
    {
      v8 = (_QWORD *)*(v5 - 1);
      v5 = &v8[(unsigned __int64)v7 / 8];
    }
    else
    {
      v8 = &v5[v7 / 0xFFFFFFFFFFFFFFF8LL];
    }
    if ( v8 != v5 )
    {
      while ( 1 )
      {
        v10 = *v8;
        if ( *(_BYTE *)(*v8 + 16LL) > 0x10u )
          goto LABEL_9;
        v11 = v21;
        if ( v22 != v21 )
          goto LABEL_8;
        v12 = &v21[HIDWORD(v23)];
        if ( v21 != v12 )
        {
          v13 = 0;
          while ( v10 != *v11 )
          {
            if ( *v11 == -2 )
              v13 = v11;
            if ( v12 == ++v11 )
            {
              if ( !v13 )
                goto LABEL_30;
              *v13 = v10;
              --v24;
              ++v20;
              goto LABEL_20;
            }
          }
          goto LABEL_9;
        }
LABEL_30:
        if ( HIDWORD(v23) < (unsigned int)v23 )
        {
          ++HIDWORD(v23);
          *v12 = v10;
          v14 = (unsigned int)v18;
          ++v20;
          if ( (unsigned int)v18 >= HIDWORD(v18) )
          {
LABEL_32:
            sub_16CD150(&v17, v19, 0, 8);
            v14 = (unsigned int)v18;
          }
LABEL_21:
          v8 += 3;
          v17[v14] = v10;
          LODWORD(v18) = v18 + 1;
          if ( v5 == v8 )
            break;
        }
        else
        {
LABEL_8:
          sub_16CCBA0(&v20, *v8);
          if ( v9 )
          {
LABEL_20:
            v14 = (unsigned int)v18;
            if ( (unsigned int)v18 >= HIDWORD(v18) )
              goto LABEL_32;
            goto LABEL_21;
          }
LABEL_9:
          v8 += 3;
          if ( v5 == v8 )
            break;
        }
      }
    }
    v15 = v17;
    v4 = v18;
    v3 = v17;
    if ( !(_DWORD)v18 )
    {
      v6 = 0;
      goto LABEL_24;
    }
  }
  v15 = v17;
LABEL_24:
  if ( v15 != v19 )
    _libc_free((unsigned __int64)v15);
  if ( v22 != v21 )
    _libc_free((unsigned __int64)v22);
  return v6;
}
