// Function: sub_2993860
// Address: 0x2993860
//
void __fastcall sub_2993860(__int64 a1, __int64 *a2, __int64 a3, char a4)
{
  __int64 v6; // rax
  __int64 v8; // rdx
  __int64 j; // r15
  __int64 v10; // r14
  unsigned __int64 v11; // rax
  int v12; // edx
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  int v18; // esi
  unsigned int v19; // edi
  __int64 v20; // rax
  __int64 v21; // r8
  __int64 v22; // rsi
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rsi
  unsigned __int64 v27; // r12
  _QWORD *v28; // rax
  __int64 v29; // r15
  __int64 v31; // [rsp+10h] [rbp-50h]
  __int64 v32; // [rsp+10h] [rbp-50h]
  __int64 i; // [rsp+18h] [rbp-48h]
  unsigned __int16 v34; // [rsp+18h] [rbp-48h]
  __int64 v35; // [rsp+20h] [rbp-40h] BYREF
  unsigned __int16 v36; // [rsp+28h] [rbp-38h]

  v6 = *a2;
  if ( (*a2 & 4) != 0 )
  {
    v8 = *(_QWORD *)(a2[4] + 16);
    for ( i = a2[4]; v8; v8 = *(_QWORD *)(v8 + 8) )
    {
      if ( (unsigned __int8)(**(_BYTE **)(v8 + 24) - 30) <= 0xAu )
        break;
    }
    v31 = 0;
    if ( v8 )
    {
      while ( 1 )
      {
        for ( j = *(_QWORD *)(v8 + 8); j; j = *(_QWORD *)(j + 8) )
        {
          if ( (unsigned __int8)(**(_BYTE **)(j + 24) - 30) <= 0xAu )
            break;
        }
        v10 = *(_QWORD *)(*(_QWORD *)(v8 + 24) + 40LL);
        if ( !(unsigned __int8)sub_22DB400(a2, v10) )
          goto LABEL_25;
        sub_2990B70(a1, v10, i);
        v11 = *(_QWORD *)(v10 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v11 == v10 + 48 )
        {
          v13 = 0;
        }
        else
        {
          if ( !v11 )
            BUG();
          v12 = *(unsigned __int8 *)(v11 - 24);
          v13 = 0;
          v14 = v11 - 24;
          if ( (unsigned int)(v12 - 30) < 0xB )
            v13 = v14;
        }
        sub_BD2ED0(v13, i, a3);
        sub_2993400(a1, v10, a3);
        if ( !a4 )
          goto LABEL_25;
        if ( !v31 )
        {
          v31 = v10;
          v8 = j;
          goto LABEL_26;
        }
        v15 = *(_QWORD *)(*(_QWORD *)(v31 + 72) + 80LL);
        if ( v15 )
        {
          v16 = v15 - 24;
          if ( v16 == v31 || v10 == v16 )
          {
            v31 = v16;
            v8 = j;
            goto LABEL_26;
          }
        }
        v17 = *(_QWORD *)(a1 + 56);
        v18 = *(_DWORD *)(v10 + 44);
        v19 = *(_DWORD *)(v17 + 32);
        v20 = (unsigned int)(*(_DWORD *)(v31 + 44) + 1);
        if ( (unsigned int)v20 >= v19 )
          break;
        v21 = *(_QWORD *)(v17 + 24);
        v22 = (unsigned int)(v18 + 1);
        v23 = 0;
        v24 = *(_QWORD *)(v21 + 8 * v20);
        if ( v19 > (unsigned int)v22 )
          goto LABEL_19;
LABEL_20:
        while ( v24 != v23 )
        {
          if ( *(_DWORD *)(v24 + 16) < *(_DWORD *)(v23 + 16) )
          {
            v25 = v24;
            v24 = v23;
            v23 = v25;
          }
          v24 = *(_QWORD *)(v24 + 8);
        }
        v31 = *(_QWORD *)v23;
LABEL_25:
        v8 = j;
LABEL_26:
        if ( !v8 )
          goto LABEL_27;
      }
      v22 = (unsigned int)(v18 + 1);
      if ( (unsigned int)v22 >= v19 )
        BUG();
      v21 = *(_QWORD *)(v17 + 24);
      v24 = 0;
LABEL_19:
      v23 = *(_QWORD *)(v21 + 8 * v22);
      goto LABEL_20;
    }
LABEL_27:
    if ( v31 )
      sub_B1AEF0(*(_QWORD *)(a1 + 56), a3, v31);
    sub_22DADE0((__int64)a2, a3);
  }
  else
  {
    v27 = v6 & 0xFFFFFFFFFFFFFFF8LL;
    sub_2990FA0(a1, v6 & 0xFFFFFFFFFFFFFFF8LL);
    sub_B43C20((__int64)&v35, v27);
    v32 = v35;
    v34 = v36;
    v28 = sub_BD2C40(72, 1u);
    v29 = (__int64)v28;
    if ( v28 )
      sub_B4C8F0((__int64)v28, a3, 1u, v32, v34);
    sub_2988C20(a1, v29, v27);
    sub_2993400(a1, v27, a3);
    if ( a4 )
      sub_B1AEF0(*(_QWORD *)(a1 + 56), a3, v27);
  }
}
