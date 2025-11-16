// Function: sub_35C9ED0
// Address: 0x35c9ed0
//
__int64 __fastcall sub_35C9ED0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 *v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  unsigned int v8; // edi
  __int64 *v9; // rsi
  __int64 *v10; // r10
  __int64 v11; // r11
  __int64 v12; // rcx
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // r11
  unsigned int v16; // ecx
  __int64 *v17; // rdx
  __int64 *v18; // rcx
  __int64 v19; // rdx
  unsigned int v20; // eax
  __int64 result; // rax

  v3 = sub_35C9B40(a2, a3, 1);
  if ( (__int64 *)v3 == v4 )
  {
LABEL_20:
    result = 0;
    if ( v6 != v5 )
      return v5;
  }
  else
  {
    v8 = *(_DWORD *)(v7 + 56);
    v9 = (__int64 *)v3;
    v10 = v4;
    while ( 1 )
    {
      v11 = *v9;
      if ( v5 )
      {
        v12 = (unsigned int)(*(_DWORD *)(v5 + 24) + 1);
        v13 = v12;
      }
      else
      {
        v12 = 0;
        v13 = 0;
      }
      v14 = 0;
      if ( v13 < v8 )
        v14 = *(__int64 **)(*(_QWORD *)(v7 + 48) + 8 * v12);
      if ( v11 )
      {
        v15 = (unsigned int)(*(_DWORD *)(v11 + 24) + 1);
        v16 = v15;
      }
      else
      {
        v15 = 0;
        v16 = 0;
      }
      v17 = 0;
      if ( v8 > v16 )
        v17 = *(__int64 **)(*(_QWORD *)(v7 + 48) + 8 * v15);
      while ( v14 != v17 )
      {
        if ( *((_DWORD *)v14 + 4) < *((_DWORD *)v17 + 4) )
        {
          v18 = v14;
          v14 = v17;
          v17 = v18;
        }
        v14 = (__int64 *)v14[1];
      }
      v5 = *v17;
      if ( *v17 )
      {
        v19 = (unsigned int)(*(_DWORD *)(v5 + 24) + 1);
        v20 = *(_DWORD *)(v5 + 24) + 1;
      }
      else
      {
        v19 = 0;
        v20 = 0;
      }
      if ( v20 >= v8 )
        BUG();
      result = **(_QWORD **)(*(_QWORD *)(v7 + 48) + 8 * v19);
      if ( !result )
        break;
      if ( v10 == ++v9 )
        goto LABEL_20;
    }
  }
  return result;
}
