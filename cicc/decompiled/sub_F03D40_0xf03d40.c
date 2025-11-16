// Function: sub_F03D40
// Address: 0xf03d40
//
unsigned __int64 *__fastcall sub_F03D40(__int64 *a1, unsigned int a2)
{
  __int64 v2; // rcx
  unsigned int v3; // r9d
  __int64 v4; // rdx
  int v5; // r10d
  __int64 v6; // r11
  unsigned __int64 *result; // rax
  int v8; // r10d
  __int64 v9; // r11
  __int64 v10; // rdx
  __int64 v11; // rbx
  __int64 v12; // rdx
  unsigned int v13; // ecx
  __int64 v14; // rax
  __int64 v15; // rax

  v2 = *a1;
  if ( a2 == 1 )
  {
    v5 = *(_DWORD *)(v2 + 12);
    v6 = *a1;
    v3 = 0;
    v4 = 0;
  }
  else
  {
    v3 = a2 - 1;
    v4 = 16LL * (a2 - 1);
    while ( 1 )
    {
      v5 = *(_DWORD *)(v2 + v4 + 12);
      v6 = v2 + v4;
      result = (unsigned __int64 *)(unsigned int)(*(_DWORD *)(v2 + v4 + 8) - 1);
      if ( (_DWORD)result != v5 )
        break;
      v4 -= 16;
      if ( !--v3 )
      {
        v5 = *(_DWORD *)(v2 + 12);
        v6 = *a1;
        v4 = 0;
        break;
      }
    }
  }
  v8 = v5 + 1;
  *(_DWORD *)(v6 + 12) = v8;
  v9 = *a1;
  v10 = *a1 + v4;
  if ( v8 != *(_DWORD *)(v10 + 8) )
  {
    v11 = v3 + 1;
    v12 = *(_QWORD *)(*(_QWORD *)v10 + 8LL * *(unsigned int *)(v10 + 12));
    if ( a2 != (_DWORD)v11 )
    {
      v13 = v3 + 1;
      while ( 1 )
      {
        v14 = v13++;
        v15 = v9 + 16 * v14;
        *(_QWORD *)v15 = v12 & 0xFFFFFFFFFFFFFFC0LL;
        *(_DWORD *)(v15 + 8) = (v12 & 0x3F) + 1;
        *(_DWORD *)(v15 + 12) = 0;
        v12 = *(_QWORD *)(v12 & 0xFFFFFFFFFFFFFFC0LL);
        if ( a2 == v13 )
          break;
        v9 = *a1;
      }
      v9 = *a1;
      v11 = a2;
    }
    result = (unsigned __int64 *)(v9 + 16 * v11);
    *result = v12 & 0xFFFFFFFFFFFFFFC0LL;
    result[1] = (unsigned int)(v12 & 0x3F) + 1;
  }
  return result;
}
