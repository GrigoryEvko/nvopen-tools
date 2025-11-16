// Function: sub_D04190
// Address: 0xd04190
//
__int64 __fastcall sub_D04190(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4, __int64 a5, __int64 *a6)
{
  __int64 v6; // r10
  __int64 v10; // rax
  __int64 v11; // rdi
  int v12; // r13d
  __int64 v14; // rdi
  __int64 v15; // rax
  unsigned int v16; // eax
  char v17; // al
  _BYTE *v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // [rsp+8h] [rbp-A8h]
  __int64 v23; // [rsp+20h] [rbp-90h] BYREF
  __int64 v24; // [rsp+28h] [rbp-88h]
  __int64 v25; // [rsp+30h] [rbp-80h]
  __int64 v26; // [rsp+38h] [rbp-78h]
  __int64 v27; // [rsp+40h] [rbp-70h]
  __int64 v28; // [rsp+48h] [rbp-68h]
  _BYTE *v29; // [rsp+50h] [rbp-60h] BYREF
  __int64 v30; // [rsp+58h] [rbp-58h]
  __int64 v31; // [rsp+60h] [rbp-50h]
  __int64 v32; // [rsp+68h] [rbp-48h]
  __int64 v33; // [rsp+70h] [rbp-40h]
  __int64 v34; // [rsp+78h] [rbp-38h]

  v6 = a5;
  if ( *a4 == 86 )
  {
    v17 = sub_D04110(a1, *(_QWORD *)(a2 - 96), *((_QWORD *)a4 - 12), (__int64)a6);
    v6 = a5;
    if ( v17 )
    {
      v18 = (_BYTE *)*((_QWORD *)a4 - 8);
      v19 = *a6;
      v30 = a5;
      v29 = v18;
      v20 = *(_QWORD *)(a2 - 64);
      v21 = a5;
      v31 = 0;
      v32 = 0;
      v33 = 0;
      v34 = 0;
      v23 = v20;
      v24 = a3;
      v25 = 0;
      v26 = 0;
      v27 = 0;
      v28 = 0;
      v12 = sub_CF4D50(v19, (__int64)&v23, (__int64)&v29, (__int64)a6, 0);
      if ( (_BYTE)v12 != 1 )
      {
        v14 = *a6;
        v29 = (_BYTE *)*((_QWORD *)a4 - 4);
        goto LABEL_5;
      }
      return 1;
    }
  }
  v10 = *(_QWORD *)(a2 - 64);
  v11 = *a6;
  v29 = a4;
  v30 = v6;
  v21 = v6;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v23 = v10;
  v24 = a3;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v12 = sub_CF4D50(v11, (__int64)&v23, (__int64)&v29, (__int64)a6, 0);
  if ( (_BYTE)v12 == 1 )
    return 1;
  v29 = a4;
  v14 = *a6;
LABEL_5:
  v15 = *(_QWORD *)(a2 - 32);
  v24 = a3;
  v30 = v21;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v23 = v15;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v16 = sub_CF4D50(v14, (__int64)&v23, (__int64)&v29, (__int64)a6, 0);
  return sub_D00090(v16, v12);
}
