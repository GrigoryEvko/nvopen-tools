// Function: sub_866420
// Address: 0x866420
//
__int64 __fastcall sub_866420(int a1)
{
  __int64 result; // rax
  int v3; // edi
  __int64 v4; // rdx
  int v5; // r9d
  __int64 *v6; // r9
  __int64 v7; // r9
  __int64 v8; // rcx
  __int64 v9; // r9
  __int64 v10; // rcx

  result = unk_4F04C48;
  if ( unk_4F04C48 != -1 )
  {
    v3 = dword_4F5FCD8;
    for ( result = qword_4F04C68[0] + 776LL * unk_4F04C48; result; result = qword_4F04C68[0] + 776 * result )
    {
      if ( *(_BYTE *)(result + 4) != 9 )
        goto LABEL_4;
      v4 = *(_QWORD *)(*(_QWORD *)(result + 408) + 72LL);
      if ( !v4 )
        goto LABEL_4;
      v5 = 0;
      do
      {
        while ( 1 )
        {
          v10 = *(_QWORD *)v4;
          if ( *(_DWORD *)(v4 + 56) >= v3 )
            break;
          v4 = *(_QWORD *)v4;
          if ( !v10 )
            goto LABEL_18;
        }
        if ( a1 )
        {
          v6 = *(__int64 **)(v4 + 8);
          if ( v6 )
            *v6 = v10;
          else
            *(_QWORD *)(*(_QWORD *)(result + 408) + 72LL) = v10;
          v7 = *(_QWORD *)(v4 + 8);
          if ( *(_QWORD *)v4 )
            *(_QWORD *)(*(_QWORD *)v4 + 8LL) = v7;
          else
            *(_QWORD *)(*(_QWORD *)(result + 408) + 80LL) = v7;
          v8 = *(_QWORD *)v4;
          v9 = qword_4F5FD38;
          *(_QWORD *)(v4 + 8) = 0;
          *(_QWORD *)v4 = v9;
          v5 = 1;
          qword_4F5FD38 = v4;
          v4 = v8;
        }
        else
        {
          *(_DWORD *)(v4 + 56) = 0;
          v5 = 1;
          v4 = v10;
        }
      }
      while ( v4 );
LABEL_18:
      if ( a1 && v5 )
      {
        *(_QWORD *)(result + 648) = *(_QWORD *)(result + 656);
        result = *(int *)(result + 552);
        if ( (_DWORD)result == -1 )
          break;
      }
      else
      {
LABEL_4:
        result = *(int *)(result + 552);
        if ( (_DWORD)result == -1 )
          break;
      }
    }
    dword_4F5FCD8 = v3 - 1;
  }
  return result;
}
