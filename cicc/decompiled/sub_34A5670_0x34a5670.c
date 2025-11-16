// Function: sub_34A5670
// Address: 0x34a5670
//
__int64 __fastcall sub_34A5670(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v9; // rdi
  __int64 v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rcx
  unsigned int v17; // eax
  __int64 result; // rax
  __int64 v19; // rdx
  __int64 v20; // rsi
  __int64 i; // rcx
  __int64 v22; // rdx
  unsigned int v23; // eax

  v7 = *(unsigned int *)(a1 + 16);
  v9 = *(_QWORD *)(a1 + 8);
  v10 = v9 + 16 * v7 - 16;
  v11 = *(_QWORD *)v10;
  if ( *(_QWORD *)(*(_QWORD *)v10 + 16LL * (unsigned int)(*(_DWORD *)(v10 + 8) - 1) + 8) >= a2 )
  {
    v22 = *(unsigned int *)(v10 + 12);
    for ( result = 16 * v22; *(_QWORD *)(v11 + 16 * v22 + 8) < a2; result = 16 * v22 )
      v22 = (unsigned int)(v22 + 1);
    *(_DWORD *)(v10 + 12) = v22;
  }
  else
  {
    *(_DWORD *)(a1 + 16) = v7 - 1;
    if ( (_DWORD)v7 == 2 )
    {
      v19 = *(_QWORD *)a1;
      LODWORD(v20) = *(_DWORD *)(v9 + 12);
    }
    else
    {
      v12 = (unsigned int)(v7 - 3);
      v13 = v9 + 16 * v12;
      if ( (_DWORD)v12 )
      {
        while ( *(_QWORD *)(*(_QWORD *)v13 + 8LL * *(unsigned int *)(v13 + 12) + 96) < a2 )
        {
          --*(_DWORD *)(a1 + 16);
          v13 -= 16;
          LODWORD(v12) = v12 - 1;
          if ( !(_DWORD)v12 )
            goto LABEL_22;
        }
        v14 = 16LL * (unsigned int)(v12 + 1) + v9;
        v15 = *(unsigned int *)(v14 + 12);
        v16 = *(_QWORD *)v14;
        v17 = *(_DWORD *)(v14 + 12);
        if ( a2 > *(_QWORD *)(*(_QWORD *)v14 + 8 * v15 + 96) )
        {
          do
            v15 = ++v17;
          while ( *(_QWORD *)(v16 + 8LL * v17 + 96) < a2 );
        }
        *(_DWORD *)(v14 + 12) = v17;
        return sub_34A3B20(a1, a2, v15, v16, a5, a6);
      }
LABEL_22:
      v19 = *(_QWORD *)a1;
      v20 = *(unsigned int *)(v9 + 12);
      if ( *(_QWORD *)(*(_QWORD *)a1 + 8 * v20 + 96) >= a2 )
      {
        v15 = *(unsigned int *)(v9 + 28);
        v16 = *(_QWORD *)(v9 + 16);
        v23 = *(_DWORD *)(v9 + 28);
        if ( *(_QWORD *)(v16 + 8 * v15 + 96) < a2 )
        {
          do
            v15 = ++v23;
          while ( *(_QWORD *)(v16 + 8LL * v23 + 96) < a2 );
        }
        *(_DWORD *)(v9 + 28) = v23;
        return sub_34A3B20(a1, a2, v15, v16, a5, a6);
      }
    }
    for ( i = *(unsigned int *)(v19 + 196); (_DWORD)i != (_DWORD)v20; LODWORD(v20) = v20 + 1 )
    {
      if ( *(_QWORD *)(v19 + 8LL * (unsigned int)v20 + 96) >= a2 )
        break;
    }
    sub_34A26E0(a1, v20, v19, i, a5, a6);
    result = *(unsigned int *)(a1 + 16);
    if ( (_DWORD)result )
    {
      result = *(_QWORD *)(a1 + 8);
      if ( *(_DWORD *)(result + 12) < *(_DWORD *)(result + 8) )
        return sub_34A3B20(a1, a2, v15, v16, a5, a6);
    }
  }
  return result;
}
