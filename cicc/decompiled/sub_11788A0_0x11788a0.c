// Function: sub_11788A0
// Address: 0x11788a0
//
_QWORD *__fastcall sub_11788A0(__int64 **a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v4; // r14
  _QWORD *v5; // r13
  __int64 *v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // r11
  __int64 v12; // r10
  unsigned int **v13; // r13
  __int64 *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  int v17; // r12d
  __int64 v18; // r11
  __int64 v19; // rdx
  int v20; // eax
  char v21; // cl
  int v22; // eax
  __int64 v23; // [rsp+0h] [rbp-90h]
  __int64 v25; // [rsp+8h] [rbp-88h]
  __int64 v26; // [rsp+10h] [rbp-80h]
  __int64 v27; // [rsp+10h] [rbp-80h]
  __int64 v28; // [rsp+18h] [rbp-78h]
  __int64 v29; // [rsp+20h] [rbp-70h] BYREF
  __int64 v30; // [rsp+28h] [rbp-68h]
  _QWORD v31[4]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v32; // [rsp+50h] [rbp-40h]

  v4 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) != 2 || a3 != v4 )
    return 0;
  v5 = *(_QWORD **)(a2 + 16);
  if ( v5 )
  {
    v5 = (_QWORD *)v5[1];
    if ( v5 )
      return 0;
    v8 = *a1;
    v9 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(*v8 + 8) + 8LL) - 17 > 1
      || (unsigned int)*(unsigned __int8 *)(v9 + 8) - 17 <= 1 )
    {
      v26 = *(_QWORD *)(a2 - 32);
      v28 = *(_QWORD *)(a2 + 72);
      v10 = sub_AD6530(v9, a3);
      v11 = v26;
      v12 = v10;
      if ( a4 )
      {
        v11 = v10;
        v12 = v26;
      }
      v23 = v11;
      v25 = v12;
      v13 = (unsigned int **)a1[1][4];
      v27 = (__int64)a1[2];
      v31[0] = sub_BD5D20(v27);
      v31[2] = ".idx";
      v14 = *a1;
      v32 = 773;
      v31[1] = v15;
      v16 = sub_B36550(v13, *v14, v23, v25, (__int64)v31, v27);
      v32 = 257;
      v29 = v16;
      v17 = sub_B4DE20(a2);
      v5 = sub_BD2C40(88, 2u);
      if ( !v5 )
        goto LABEL_14;
      v18 = *(_QWORD *)(v4 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v18 + 8) - 17 <= 1 )
      {
LABEL_13:
        sub_B44260((__int64)v5, v18, 34, 2u, 0, 0);
        v5[9] = v28;
        v5[10] = sub_B4DC50(v28, (__int64)&v29, 1);
        sub_B4D9A0((__int64)v5, v4, &v29, 1, (__int64)v31);
LABEL_14:
        sub_B4DDE0((__int64)v5, v17);
        return v5;
      }
      v19 = *(_QWORD *)(v29 + 8);
      v20 = *(unsigned __int8 *)(v19 + 8);
      if ( v20 == 17 )
      {
        v21 = 0;
      }
      else
      {
        v21 = 1;
        if ( v20 != 18 )
          goto LABEL_13;
      }
      v22 = *(_DWORD *)(v19 + 32);
      BYTE4(v30) = v21;
      LODWORD(v30) = v22;
      v18 = sub_BCE1B0((__int64 *)v18, v30);
      goto LABEL_13;
    }
  }
  return v5;
}
