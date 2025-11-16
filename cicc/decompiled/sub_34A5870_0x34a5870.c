// Function: sub_34A5870
// Address: 0x34a5870
//
unsigned __int64 __fastcall sub_34A5870(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 result; // rax
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rsi
  unsigned __int64 v10; // rsi
  __int64 v11; // rdx
  unsigned __int64 *v12; // r13
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  unsigned __int64 *v17; // r13
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  unsigned __int64 *v22; // r13
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  unsigned __int64 *v27; // r13

  result = *(unsigned int *)(a1 + 16);
  if ( (_DWORD)result )
  {
    v7 = *(_QWORD *)(a1 + 8);
    if ( *(_DWORD *)(v7 + 12) < *(_DWORD *)(v7 + 8) )
    {
      v8 = *(unsigned int *)(a1 + 104);
      if ( (_DWORD)v8 )
      {
        v9 = *(_QWORD *)(a1 + 96);
        if ( *(_DWORD *)(v9 + 12) < *(_DWORD *)(v9 + 8) )
        {
          v10 = *(_QWORD *)(*(_QWORD *)(v9 + 16 * v8 - 16) + 16LL * *(unsigned int *)(v9 + 16 * v8 - 16 + 12));
          v11 = v7 + 16 * result - 16;
          if ( v10 > *(_QWORD *)(*(_QWORD *)v11 + 16LL * *(unsigned int *)(v11 + 12) + 8) )
          {
            result = sub_34A5800(a1, v10, v11, v7, a5, a6);
            if ( !*(_DWORD *)(a1 + 16) )
              return result;
            result = *(_QWORD *)(a1 + 8);
            if ( *(_DWORD *)(result + 12) >= *(_DWORD *)(result + 8) )
              return result;
            v27 = (unsigned __int64 *)sub_34A2590(a1);
            result = *(_QWORD *)sub_34A25B0(a1 + 88);
            if ( *v27 <= result )
              return result;
            v17 = (unsigned __int64 *)sub_34A2590(a1 + 88);
          }
          else
          {
            v12 = (unsigned __int64 *)sub_34A2590(a1);
            result = sub_34A25B0(a1 + 88);
            if ( *v12 <= *(_QWORD *)result )
              return result;
            result = sub_34A5800(a1 + 88, *v12, v13, v14, v15, v16);
            if ( !*(_DWORD *)(a1 + 104) )
              return result;
            result = *(_QWORD *)(a1 + 96);
            if ( *(_DWORD *)(result + 12) >= *(_DWORD *)(result + 8) )
              return result;
            v17 = (unsigned __int64 *)sub_34A2590(a1 + 88);
            result = *(_QWORD *)sub_34A25B0(a1);
            if ( *v17 <= result )
              return result;
          }
          do
          {
            result = sub_34A5800(a1, *v17, v18, v19, v20, v21);
            if ( !*(_DWORD *)(a1 + 16) )
              break;
            result = *(_QWORD *)(a1 + 8);
            if ( *(_DWORD *)(result + 12) >= *(_DWORD *)(result + 8) )
              break;
            v22 = (unsigned __int64 *)sub_34A2590(a1);
            result = sub_34A25B0(a1 + 88);
            if ( *(_QWORD *)result >= *v22 )
              break;
            sub_34A5800(a1 + 88, *v22, v23, v24, v25, v26);
            result = *(unsigned int *)(a1 + 104);
            if ( !(_DWORD)result )
              break;
            result = *(_QWORD *)(a1 + 96);
            if ( *(_DWORD *)(result + 12) >= *(_DWORD *)(result + 8) )
              break;
            v17 = (unsigned __int64 *)sub_34A2590(a1 + 88);
            result = sub_34A25B0(a1);
            v19 = *v17;
          }
          while ( *(_QWORD *)result < *v17 );
        }
      }
    }
  }
  return result;
}
