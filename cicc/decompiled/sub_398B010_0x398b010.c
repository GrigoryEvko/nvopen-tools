// Function: sub_398B010
// Address: 0x398b010
//
_QWORD *__fastcall sub_398B010(__int64 a1)
{
  _QWORD *result; // rax
  __int64 v3; // rdi
  __int64 v4; // r12
  void (__fastcall *v5)(__int64, _QWORD, _QWORD); // r13
  __int64 v6; // rax
  __int64 v7; // rdi
  int v8; // r12d
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 *v14; // r15
  __int64 v15; // rsi
  _QWORD *v16; // [rsp-50h] [rbp-50h]
  _QWORD *v17; // [rsp-48h] [rbp-48h]
  __int64 *v18; // [rsp-40h] [rbp-40h]

  result = (_QWORD *)*(unsigned int *)(a1 + 1200);
  if ( (_DWORD)result )
  {
    v3 = *(_QWORD *)(a1 + 8);
    v4 = *(_QWORD *)(v3 + 256);
    v5 = *(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v4 + 160LL);
    v6 = sub_396DD80(v3);
    v5(v4, *(_QWORD *)(v6 + 144), 0);
    v7 = *(_QWORD *)(a1 + 8);
    v17 = *(_QWORD **)(a1 + 1192);
    v8 = *(_DWORD *)(*(_QWORD *)(v7 + 240) + 8LL);
    result = &v17[4 * *(unsigned int *)(a1 + 1200)];
    v16 = result;
    if ( v17 != result )
    {
      while ( 1 )
      {
        if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(v7 + 232) + 504LL) - 34) > 1 )
        {
          (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(v7 + 256) + 176LL))(
            *(_QWORD *)(v7 + 256),
            v17[1],
            0);
          v7 = *(_QWORD *)(a1 + 8);
        }
        v9 = *v17;
        v10 = v17[2];
        if ( (((__int64)v17 - *(_QWORD *)(a1 + 1192)) >> 5) + 1 == *(_DWORD *)(a1 + 1200) )
          v11 = *(unsigned int *)(a1 + 1344);
        else
          v11 = v17[6];
        v12 = 32 * (v11 - v10);
        v13 = *(_QWORD *)(a1 + 1336) + 32 * v10;
        v18 = (__int64 *)(v13 + v12);
        if ( v13 + v12 != v13 )
        {
          v14 = (__int64 *)v13;
          do
          {
            if ( *(_QWORD *)(v9 + 856) )
            {
              sub_396F380(v7);
              sub_396F380(*(_QWORD *)(a1 + 8));
            }
            else
            {
              sub_38DDC80(*(__int64 **)(v7 + 256), *v14, (unsigned __int8)v8, 0);
              sub_38DDC80(*(__int64 **)(*(_QWORD *)(a1 + 8) + 256LL), v14[1], (unsigned __int8)v8, 0);
            }
            v15 = (__int64)v14;
            v14 += 4;
            sub_398AF30(a1, v15);
            v7 = *(_QWORD *)(a1 + 8);
          }
          while ( v18 != v14 );
        }
        (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(v7 + 256) + 424LL))(
          *(_QWORD *)(v7 + 256),
          0,
          (unsigned __int8)v8);
        (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 256LL) + 424LL))(
          *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL),
          0,
          (unsigned __int8)v8);
        v17 += 4;
        result = v17;
        if ( v16 == v17 )
          break;
        v7 = *(_QWORD *)(a1 + 8);
      }
    }
  }
  return result;
}
