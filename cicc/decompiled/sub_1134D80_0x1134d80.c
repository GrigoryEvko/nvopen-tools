// Function: sub_1134D80
// Address: 0x1134d80
//
__int64 __fastcall sub_1134D80(__int64 a1, __int64 a2)
{
  __int64 *v4; // rax
  _QWORD *v5; // rsi
  unsigned int v6; // r15d
  _BYTE *v7; // r12
  _BYTE *v8; // r14
  __int64 v9; // rax
  unsigned int v10; // eax
  __int64 v11; // r13
  _QWORD v13[2]; // [rsp+10h] [rbp-F0h] BYREF
  unsigned __int8 v14; // [rsp+20h] [rbp-E0h]
  __int64 v15; // [rsp+28h] [rbp-D8h]
  __int64 v16; // [rsp+30h] [rbp-D0h]
  __int64 v17; // [rsp+38h] [rbp-C8h] BYREF
  unsigned int v18; // [rsp+40h] [rbp-C0h]
  _BYTE *v19; // [rsp+78h] [rbp-88h] BYREF
  __int64 v20; // [rsp+80h] [rbp-80h]
  _BYTE v21[120]; // [rsp+88h] [rbp-78h] BYREF

  v13[1] = a2;
  v14 = 0;
  v15 = 0;
  v16 = 1;
  v13[0] = off_49E62C8;
  v4 = &v17;
  do
  {
    *v4 = -4096;
    v4 += 2;
  }
  while ( v4 != (__int64 *)&v19 );
  v5 = v13;
  v19 = v21;
  v20 = 0x400000000LL;
  sub_D13D60(a2, (__int64)v13, 0);
  v6 = v14;
  if ( v14 )
  {
    v7 = v19;
    v6 = 0;
  }
  else
  {
    v7 = &v19[16 * (unsigned int)v20];
    if ( v19 != v7 )
    {
      v8 = v19;
      do
      {
        v10 = *((_DWORD *)v8 + 2);
        v11 = *(_QWORD *)v8;
        if ( v10 <= 2 )
        {
          if ( !v10 )
            goto LABEL_20;
          v6 = 1;
          v9 = sub_AD64C0(*(_QWORD *)(v11 + 8), (*(_WORD *)(v11 + 2) & 0x3F) == 33, 0);
          sub_F162A0(a1, v11, v9);
          v5 = (_QWORD *)v11;
          sub_F207A0(a1, (__int64 *)v11);
        }
        else if ( v10 != 3 )
        {
LABEL_20:
          BUG();
        }
        v8 += 16;
      }
      while ( v7 != v8 );
      v7 = v19;
    }
  }
  v13[0] = off_49E62C8;
  if ( v7 != v21 )
    _libc_free(v7, v5);
  if ( (v16 & 1) == 0 )
    sub_C7D6A0(v17, 16LL * v18, 8);
  nullsub_185();
  return v6;
}
