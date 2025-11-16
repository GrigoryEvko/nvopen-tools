// Function: sub_39D3860
// Address: 0x39d3860
//
__int64 __fastcall sub_39D3860(__int64 *a1, int a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // r8
  unsigned int v5; // ecx
  int *v6; // rax
  int v7; // r9d
  int v9; // eax
  int v10; // r11d

  v2 = a1[3];
  v3 = *(unsigned int *)(v2 + 24);
  v4 = *(_QWORD *)(v2 + 8);
  if ( !(_DWORD)v3 )
  {
LABEL_6:
    v6 = (int *)(v4 + 48 * v3);
    return sub_1E31F40(*a1, v6[10], *((_BYTE *)v6 + 44), *((char **)v6 + 1), *((_QWORD *)v6 + 2));
  }
  v5 = (v3 - 1) & (37 * a2);
  v6 = (int *)(v4 + 48LL * v5);
  v7 = *v6;
  if ( *v6 != a2 )
  {
    v9 = 1;
    while ( v7 != 0x7FFFFFFF )
    {
      v10 = v9 + 1;
      v5 = (v3 - 1) & (v9 + v5);
      v6 = (int *)(v4 + 48LL * v5);
      v7 = *v6;
      if ( *v6 == a2 )
        return sub_1E31F40(*a1, v6[10], *((_BYTE *)v6 + 44), *((char **)v6 + 1), *((_QWORD *)v6 + 2));
      v9 = v10;
    }
    goto LABEL_6;
  }
  return sub_1E31F40(*a1, v6[10], *((_BYTE *)v6 + 44), *((char **)v6 + 1), *((_QWORD *)v6 + 2));
}
