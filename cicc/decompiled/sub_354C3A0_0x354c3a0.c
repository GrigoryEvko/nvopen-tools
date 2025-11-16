// Function: sub_354C3A0
// Address: 0x354c3a0
//
__int64 __fastcall sub_354C3A0(__int64 a1, __int64 *a2, int a3, int a4)
{
  int v5; // r12d
  char v6; // r15
  __int64 *v7; // rsi
  int v8; // eax
  void (*v9)(void); // rax
  __int64 (*v10)(); // rax
  unsigned __int64 *v11; // rdi
  _QWORD *v12; // rax
  __int64 *v13; // r13
  int v14; // r14d
  __int64 v15; // rax
  unsigned __int64 v16; // r12
  _QWORD *v17; // rax
  _QWORD *v18; // rdx
  _QWORD *v19; // rcx
  char v20; // di
  int v21; // eax
  __int64 *v23; // [rsp+8h] [rbp-48h] BYREF
  int v24[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v23 = a2;
  if ( a3 > a4 )
  {
    v5 = a4 - 1;
    v6 = 0;
  }
  else
  {
    v5 = a4 + 1;
    v6 = 1;
  }
  v24[0] = a3;
  if ( v5 == a3 )
    return 0;
  while ( 1 )
  {
    v9 = *(void (**)(void))(**(_QWORD **)(a1 + 96) + 128LL);
    if ( (char *)v9 == (char *)sub_2DAC790 )
    {
      v7 = v23;
      if ( *(_WORD *)(*v23 + 68) <= 0x14u )
        goto LABEL_12;
    }
    else
    {
      v9();
      v7 = v23;
      if ( *(_WORD *)(*v23 + 68) <= 0x14u )
      {
LABEL_12:
        v10 = *(__int64 (**)())(**(_QWORD **)(a1 + 96) + 128LL);
        if ( v10 == sub_2DAC790 )
          goto LABEL_13;
        goto LABEL_33;
      }
    }
    if ( (unsigned __int8)sub_3545540(a1 + 112, v7, v24[0]) )
      break;
    v8 = v24[0] + 1;
    if ( !v6 )
      v8 = v24[0] - 1;
    v24[0] = v8;
    if ( v5 == v8 )
      return 0;
  }
  v10 = *(__int64 (**)())(**(_QWORD **)(a1 + 96) + 128LL);
  if ( v10 == sub_2DAC790 )
    goto LABEL_28;
LABEL_33:
  v10();
LABEL_28:
  if ( *(_WORD *)(*v23 + 68) > 0x14u )
    sub_35452D0(a1 + 112, v23, v24[0]);
LABEL_13:
  v11 = (unsigned __int64 *)sub_354BE50(a1, v24);
  v12 = (_QWORD *)v11[6];
  if ( v12 == (_QWORD *)(v11[8] - 8) )
  {
    sub_354B0D0(v11, &v23);
    v13 = v23;
  }
  else
  {
    v13 = v23;
    if ( v12 )
    {
      *v12 = v23;
      v12 = (_QWORD *)v11[6];
    }
    v11[6] = (unsigned __int64)(v12 + 1);
  }
  v14 = v24[0];
  v15 = sub_22077B0(0x30u);
  *(_QWORD *)(v15 + 32) = v13;
  v16 = v15;
  *(_DWORD *)(v15 + 40) = v14;
  v17 = sub_3549C20(a1 + 32, (unsigned __int64 *)(v15 + 32));
  if ( v18 )
  {
    v19 = (_QWORD *)(a1 + 40);
    v20 = 1;
    if ( !v17 && v19 != v18 )
      v20 = (unsigned __int64)v13 < v18[4];
    sub_220F040(v20, v16, v18, v19);
    ++*(_QWORD *)(a1 + 72);
  }
  else
  {
    j_j___libc_free_0(v16);
  }
  v21 = v24[0];
  if ( *(_DWORD *)(a1 + 84) < v24[0] )
    *(_DWORD *)(a1 + 84) = v24[0];
  if ( v21 < *(_DWORD *)(a1 + 80) )
    *(_DWORD *)(a1 + 80) = v21;
  return 1;
}
