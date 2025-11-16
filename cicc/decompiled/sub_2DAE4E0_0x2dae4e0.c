// Function: sub_2DAE4E0
// Address: 0x2dae4e0
//
__int64 __fastcall sub_2DAE4E0(_QWORD *a1, __int64 a2, int a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 v8; // rdx
  __int64 v9; // r9
  __int64 v10; // r12
  __int64 v12; // rdx
  _QWORD *v13; // rdi
  __int64 v14; // r15
  __int64 v15; // rax
  _QWORD *v17; // r10
  __int64 v18; // rsi
  _QWORD *v19; // r15
  __int64 v20; // rsi
  __int64 v21; // rax

  v5 = *(_QWORD *)(a2 + 16);
  v8 = *(unsigned __int16 *)(v5 + 68);
  v9 = a4;
  v10 = a4;
  switch ( *(_WORD *)(v5 + 68) )
  {
    case 0:
    case 0x14:
      return v10 & sub_2EBF1E0(*a1, *(unsigned int *)(a2 + 8), v8, a4, a5, v9);
    case 8:
      v20 = *(_QWORD *)(*(_QWORD *)(v5 + 32) + 104LL);
      if ( (_DWORD)v20 )
        v10 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(*(_QWORD *)a1[1] + 320LL))(
                a1[1],
                v20,
                a4,
                a5);
      break;
    case 9:
      v17 = (_QWORD *)a1[1];
      v18 = *(_QWORD *)(*(_QWORD *)(v5 + 32) + 144LL);
      if ( a3 == 2 )
      {
        if ( (_DWORD)v18 )
        {
          v21 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(*v17 + 312LL))(a1[1], v18, a4, a5);
          v17 = (_QWORD *)a1[1];
          v10 = v21;
        }
        v10 &= *(_QWORD *)(v17[34] + 16LL * (unsigned int)v18);
      }
      else
      {
        v19 = (_QWORD *)(v17[34] + 16LL * (unsigned int)v18);
        v8 = ~v19[1];
        v9 = ~*v19 & a4;
        v10 = v9;
      }
      break;
    case 0x13:
      v12 = (unsigned int)(a3 + 1);
      a4 = *(_QWORD *)(v5 + 32);
      v13 = (_QWORD *)a1[1];
      v8 = a4 + 40 * v12;
      v14 = *(_QWORD *)(v8 + 24);
      if ( (_DWORD)v14 )
      {
        v15 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD, __int64, __int64))(*v13 + 312LL))(
                v13,
                (unsigned int)v14,
                v9,
                a5);
        v13 = (_QWORD *)a1[1];
        v10 = v15;
      }
      v10 &= *(_QWORD *)(v13[34] + 16LL * (unsigned int)v14);
      break;
    default:
      BUG();
  }
  return v10 & sub_2EBF1E0(*a1, *(unsigned int *)(a2 + 8), v8, a4, a5, v9);
}
