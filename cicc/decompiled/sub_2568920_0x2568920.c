// Function: sub_2568920
// Address: 0x2568920
//
__int64 __fastcall sub_2568920(__int64 a1, __int64 a2, unsigned __int8 (__fastcall *a3)(__int64), __int64 a4)
{
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // r15
  __int64 v11; // rsi
  size_t v13; // r15
  __int64 v14; // [rsp+8h] [rbp-88h]
  void *v15; // [rsp+10h] [rbp-80h]
  __int64 v16; // [rsp+18h] [rbp-78h]
  __int64 v17; // [rsp+20h] [rbp-70h] BYREF
  void *v18; // [rsp+28h] [rbp-68h]
  __int64 v19; // [rsp+30h] [rbp-60h]
  __int64 v20; // [rsp+38h] [rbp-58h]
  __int64 v21; // [rsp+40h] [rbp-50h]
  __int64 v22; // [rsp+48h] [rbp-48h]
  __int64 v23; // [rsp+50h] [rbp-40h]
  __int64 v24; // [rsp+58h] [rbp-38h]

  v6 = sub_2568740(a1, a2);
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  sub_C7D6A0(0, 0, 8);
  v7 = *(unsigned int *)(v6 + 24);
  LODWORD(v20) = v7;
  if ( (_DWORD)v7 )
  {
    v18 = (void *)sub_C7D670(8 * v7, 8);
    v19 = *(_QWORD *)(v6 + 16);
    memcpy(v18, *(const void **)(v6 + 8), 8LL * (unsigned int)v20);
  }
  else
  {
    v18 = 0;
    v19 = 0;
  }
  v21 = *(_QWORD *)(v6 + 32);
  v22 = *(_QWORD *)(v6 + 40);
  v23 = *(_QWORD *)(v6 + 48);
  v24 = *(_QWORD *)(v6 + 56);
  sub_C7D6A0(0, 0, 8);
  v8 = *(unsigned int *)(a1 + 224);
  if ( (_DWORD)v8 )
  {
    v13 = 8 * v8;
    v14 = 8 * v8;
    v15 = (void *)sub_C7D670(8 * v8, 8);
    memcpy(v15, *(const void **)(a1 + 208), v13);
  }
  else
  {
    v14 = 0;
    v15 = 0;
  }
  v9 = *(_QWORD *)(a1 + 240);
  v10 = *(_QWORD *)(a1 + 248);
  v11 = v22;
  v16 = *(_QWORD *)(a1 + 256);
  while ( (v9 != v11 || v23 != v10 || v24 != v16) && a3(a4) )
  {
    v22 = sub_3106C80(&v17);
    v11 = v22;
  }
  sub_C7D6A0((__int64)v15, v14, 8);
  return sub_C7D6A0((__int64)v18, 8LL * (unsigned int)v20, 8);
}
