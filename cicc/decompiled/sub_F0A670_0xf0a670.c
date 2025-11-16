// Function: sub_F0A670
// Address: 0xf0a670
//
_QWORD *__fastcall sub_F0A670(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  _QWORD *v12; // r14
  __int64 v13; // r11
  __int64 v14; // r8
  unsigned __int16 v15; // r9
  unsigned int v16; // ecx
  __int64 *v18; // rdi
  __int64 *v19; // rax
  __int64 v20; // rsi
  int v21; // edx
  char v22; // dl
  __int64 v23; // rax
  int v24; // [rsp+8h] [rbp-58h]
  __int64 v26; // [rsp+28h] [rbp-38h]

  v24 = a4 + 1;
  v12 = sub_BD2C40(88, (int)a4 + 1);
  if ( v12 )
  {
    v13 = *(_QWORD *)(a2 + 8);
    v14 = a7;
    v15 = a8;
    v16 = v24 & 0x7FFFFFF;
    if ( (unsigned int)*(unsigned __int8 *)(v13 + 8) - 17 > 1 )
    {
      v18 = &a3[a4];
      if ( v18 != a3 )
      {
        v19 = a3;
        v20 = *(_QWORD *)(*a3 + 8);
        v21 = *(unsigned __int8 *)(v20 + 8);
        if ( v21 == 17 )
        {
LABEL_10:
          v22 = 0;
        }
        else
        {
          while ( v21 != 18 )
          {
            if ( v18 == ++v19 )
              goto LABEL_3;
            v20 = *(_QWORD *)(*v19 + 8);
            v21 = *(unsigned __int8 *)(v20 + 8);
            if ( v21 == 17 )
              goto LABEL_10;
          }
          v22 = 1;
        }
        BYTE4(v26) = v22;
        LODWORD(v26) = *(_DWORD *)(v20 + 32);
        v23 = sub_BCE1B0((__int64 *)v13, v26);
        v16 = v24 & 0x7FFFFFF;
        v15 = a8;
        v14 = a7;
        v13 = v23;
      }
    }
LABEL_3:
    sub_B44260((__int64)v12, v13, 34, v16, v14, v15);
    v12[9] = a1;
    v12[10] = sub_B4DC50(a1, (__int64)a3, a4);
    sub_B4D9A0((__int64)v12, a2, a3, a4, a5);
  }
  sub_B4DDE0((__int64)v12, 3);
  return v12;
}
