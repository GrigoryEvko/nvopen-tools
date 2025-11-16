// Function: sub_9C6B10
// Address: 0x9c6b10
//
__int64 __fastcall sub_9C6B10(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v12; // r14
  __int64 v13; // r11
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rcx
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rsi
  int v21; // edx
  char v22; // dl
  __int64 v23; // rax
  int v24; // [rsp+8h] [rbp-58h]
  __int64 v26; // [rsp+28h] [rbp-38h]

  v24 = a4 + 1;
  v12 = sub_BD2C40(88, (unsigned int)(a4 + 1));
  if ( v12 )
  {
    v13 = *(_QWORD *)(a2 + 8);
    v14 = a7;
    v15 = (unsigned __int16)a8;
    v16 = v24 & 0x7FFFFFF;
    if ( (unsigned int)*(unsigned __int8 *)(v13 + 8) - 17 > 1 )
    {
      v18 = a3 + 8 * a4;
      if ( v18 != a3 )
      {
        v19 = a3;
        v20 = *(_QWORD *)(*(_QWORD *)a3 + 8LL);
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
            v19 += 8;
            if ( v18 == v19 )
              goto LABEL_3;
            v20 = *(_QWORD *)(*(_QWORD *)v19 + 8LL);
            v21 = *(unsigned __int8 *)(v20 + 8);
            if ( v21 == 17 )
              goto LABEL_10;
          }
          v22 = 1;
        }
        BYTE4(v26) = v22;
        LODWORD(v26) = *(_DWORD *)(v20 + 32);
        v23 = sub_BCE1B0(v13, v26);
        v16 = v24 & 0x7FFFFFF;
        v15 = (unsigned __int16)a8;
        v14 = a7;
        v13 = v23;
      }
    }
LABEL_3:
    sub_B44260(v12, v13, 34, v16, v14, v15);
    *(_QWORD *)(v12 + 72) = a1;
    *(_QWORD *)(v12 + 80) = sub_B4DC50(a1, a3, a4);
    sub_B4D9A0(v12, a2, a3, a4, a5);
  }
  return v12;
}
