// Function: sub_2F74980
// Address: 0x2f74980
//
__int64 __fastcall sub_2F74980(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 result; // rax
  _QWORD *v12; // r14
  __int64 v13; // rdi
  __int64 v14; // r14
  unsigned int v15; // r13d
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9

  if ( (int)a2 >= 0 )
  {
    v12 = *(_QWORD **)(a1 + 16);
    result = *(_QWORD *)(*(_QWORD *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*v12 + 16LL) + 200LL))(*(_QWORD *)(*v12 + 16LL))
                                   + 248)
                       + 16LL);
    if ( *(_BYTE *)(result + (unsigned int)a2) )
    {
      result = *(_QWORD *)(v12[48] + 8LL * ((unsigned int)a2 >> 6)) & (1LL << a2);
      if ( !result )
      {
        v13 = *(_QWORD *)(a1 + 8);
        result = *(_DWORD *)(*(_QWORD *)(v13 + 8) + 24LL * (unsigned int)a2 + 16) >> 12;
        v14 = *(_QWORD *)(v13 + 56) + 2 * result;
        v15 = *(_DWORD *)(*(_QWORD *)(v13 + 8) + 24LL * (unsigned int)a2 + 16) & 0xFFF;
        if ( v14 )
        {
          while ( 1 )
          {
            v14 += 2;
            v16 = sub_2FF6500(v13, v15, 1);
            sub_2F747D0(
              a4,
              v15,
              *(_QWORD *)(v16 + 24),
              v17,
              v18,
              v19,
              v15,
              *(_QWORD *)(v16 + 24),
              *(_QWORD *)(v16 + 32));
            result = (unsigned int)*(__int16 *)(v14 - 2);
            if ( !*(_WORD *)(v14 - 2) )
              break;
            v13 = *(_QWORD *)(a1 + 8);
            v15 += result;
          }
        }
      }
    }
  }
  else
  {
    if ( a3 )
    {
      v8 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 272LL) + 16LL * a3);
      v9 = *v8;
      v10 = v8[1];
    }
    else
    {
      v9 = sub_2EBF1E0(*(_QWORD *)(a1 + 16), a2);
    }
    return (__int64)sub_2F747D0(a4, a2, v10, a4, a5, a6, a2, v9, v10);
  }
  return result;
}
