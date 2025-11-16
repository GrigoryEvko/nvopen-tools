// Function: sub_1E47570
// Address: 0x1e47570
//
__int64 __fastcall sub_1E47570(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // r12
  char v8; // r13
  unsigned int v9; // ebx
  unsigned int v10; // r14d
  unsigned int v11; // eax
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // r15
  unsigned int v16; // r12d
  __int64 v17; // rbx
  int v18; // r14d
  unsigned int v19; // r13d
  unsigned int v20; // eax
  unsigned int v21; // ebx
  bool v23; // al
  _QWORD *v24; // rdx
  __int64 v28; // [rsp+18h] [rbp-78h]
  __int64 v29; // [rsp+20h] [rbp-70h]
  unsigned int v30; // [rsp+20h] [rbp-70h]
  __int64 v31; // [rsp+28h] [rbp-68h]
  __int64 v32; // [rsp+38h] [rbp-58h]
  __int64 v33; // [rsp+40h] [rbp-50h]
  int v34; // [rsp+4Ch] [rbp-44h]
  char v35; // [rsp+4Ch] [rbp-44h]
  int v36; // [rsp+50h] [rbp-40h] BYREF
  int v37; // [rsp+54h] [rbp-3Ch] BYREF
  _QWORD v38[7]; // [rsp+58h] [rbp-38h] BYREF

  v31 = a2;
  v33 = (a2 - 1) / 2;
  if ( a2 <= a3 )
  {
    v32 = a1 + 8 * a2;
  }
  else
  {
    while ( 1 )
    {
      v36 = 0;
      v37 = 0;
      v32 = a1 + 8 * v33;
      v29 = *(_QWORD *)(*(_QWORD *)a5 + 96LL);
      v5 = v29 + 10LL * *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v32 + 16LL) + 6LL);
      v28 = *(_QWORD *)(*(_QWORD *)a5 + 72LL);
      v6 = v28 + 16LL * *(unsigned __int16 *)(v5 + 2);
      v7 = v28 + 16LL * *(unsigned __int16 *)(v5 + 4);
      if ( v6 == v7 )
      {
        v9 = -1;
      }
      else
      {
        v34 = 0;
        v8 = 0;
        v9 = -1;
        do
        {
          v10 = *(_DWORD *)(v6 + 4);
          v11 = sub_39FAC40(v10);
          if ( v11 < v9 )
          {
            v34 = v10;
            v9 = v11;
            v8 = 1;
          }
          v6 += 16;
        }
        while ( v7 != v6 );
        if ( v8 )
          v36 = v34;
      }
      v12 = v29 + 10LL * *(unsigned __int16 *)(*(_QWORD *)(a4 + 16) + 6LL);
      v13 = 16LL * *(unsigned __int16 *)(v12 + 2);
      v14 = 16LL * *(unsigned __int16 *)(v12 + 4);
      v15 = v28 + v14;
      if ( v28 + v13 == v28 + v14 )
      {
        v32 = a1 + 8 * v31;
        goto LABEL_22;
      }
      v35 = 0;
      v16 = -1;
      v30 = v9;
      v17 = v28 + v13;
      v18 = 0;
      do
      {
        v19 = *(_DWORD *)(v17 + 4);
        v20 = sub_39FAC40(v19);
        if ( v20 < v16 )
        {
          v35 = 1;
          v18 = v19;
          v16 = v20;
        }
        v17 += 16;
      }
      while ( v15 != v17 );
      if ( v35 )
        v37 = v18;
      if ( v30 == 1 && v16 == 1 )
      {
        v21 = 0;
        if ( (unsigned __int8)sub_1932870(a5 + 8, &v36, v38) )
          v21 = *(_DWORD *)(v38[0] + 4LL);
        if ( !(unsigned __int8)sub_1932870(a5 + 8, &v37, v38) )
        {
          v32 = a1 + 8 * v31;
          goto LABEL_22;
        }
        v23 = *(_DWORD *)(v38[0] + 4LL) > v21;
      }
      else
      {
        v23 = v30 > v16;
      }
      v24 = (_QWORD *)(a1 + 8 * v31);
      if ( !v23 )
        break;
      v31 = v33;
      *v24 = *(_QWORD *)v32;
      if ( a3 >= v33 )
        goto LABEL_22;
      v33 = (v33 - 1) / 2;
    }
    v32 = a1 + 8 * v31;
  }
LABEL_22:
  *(_QWORD *)v32 = a4;
  return v32;
}
