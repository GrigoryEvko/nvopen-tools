// Function: sub_10997E0
// Address: 0x10997e0
//
bool __fastcall sub_10997E0(_QWORD *a1, _BYTE *a2, __int64 a3)
{
  _BYTE *v3; // r8
  __int64 v4; // r13
  __int64 v5; // r15
  _BYTE *v7; // rdx
  _BYTE *v8; // r9
  __int64 v9; // r11
  __int64 v10; // rdi
  _BYTE *v11; // rax
  char v12; // bl
  unsigned __int8 v13; // cl
  _QWORD *v14; // r9
  unsigned __int64 v15; // rdx
  __int64 v17; // [rsp+8h] [rbp-48h]
  _QWORD v18[8]; // [rsp+10h] [rbp-40h] BYREF

  v3 = &a2[a3];
  v4 = a1[2];
  v5 = a1[3];
  if ( &a2[a3] != a2 )
  {
    v17 = a1[3];
    v7 = a2;
    v8 = (_BYTE *)a1[2];
    v9 = 0;
    v10 = 0;
    v11 = 0;
    while ( 1 )
    {
      if ( (_BYTE *)(v4 + v5) == v8 )
        goto LABEL_6;
      v12 = *v8;
      if ( *v8 == 42 )
      {
        ++v8;
        v9 = v10;
        v7 = a2;
        v11 = v8;
        goto LABEL_8;
      }
      v13 = *a2;
      if ( v12 != 91 )
      {
        if ( v12 == 92 )
        {
          if ( v8[1] == v13 )
          {
            v8 += 2;
            ++a2;
            goto LABEL_8;
          }
        }
        else if ( v12 == v13 || v12 == 63 )
        {
          ++v8;
          ++a2;
          goto LABEL_8;
        }
        goto LABEL_6;
      }
      v14 = (_QWORD *)(*a1 + 80 * v10);
      if ( (*(_QWORD *)(v14[1] + 8LL * (v13 >> 6)) & (1LL << v13)) == 0 )
      {
LABEL_6:
        if ( !v11 )
          return 0;
        a2 = v7 + 1;
        v10 = v9;
        v8 = v11;
        ++v7;
LABEL_8:
        if ( v3 == a2 )
          goto LABEL_14;
      }
      else
      {
        ++a2;
        ++v10;
        v8 = (_BYTE *)(v4 + *v14);
        if ( v3 == a2 )
        {
LABEL_14:
          v5 = v17;
          v15 = (unsigned __int64)&v8[-v4];
          goto LABEL_15;
        }
      }
    }
  }
  v15 = 0;
LABEL_15:
  v18[0] = v4;
  v18[1] = v5;
  return sub_C93580(v18, 42, v15) == -1;
}
