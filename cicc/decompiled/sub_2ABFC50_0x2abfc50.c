// Function: sub_2ABFC50
// Address: 0x2abfc50
//
__int64 __fastcall sub_2ABFC50(__int64 a1, int *a2, _QWORD *a3)
{
  int v4; // edx
  int v5; // r8d
  int v6; // ebx
  char v7; // r9
  __int64 v8; // r10
  int v9; // edx
  __int64 v10; // r11
  unsigned int i; // eax
  int *v12; // rdi
  int v13; // esi
  unsigned int v14; // eax

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v5 = *a2;
  v6 = 1;
  v7 = *((_BYTE *)a2 + 4);
  v8 = *(_QWORD *)(a1 + 8);
  v9 = v4 - 1;
  v10 = 0;
  for ( i = v9 & ((v7 == 0) + 37 * *a2 - 1); ; i = v9 & v14 )
  {
    v12 = (int *)(v8 + 72LL * i);
    v13 = *v12;
    if ( *v12 == v5 && v7 == *((_BYTE *)v12 + 4) )
    {
      *a3 = v12;
      return 1;
    }
    if ( v13 == -1 )
      break;
    if ( v13 == -2 && *((_BYTE *)v12 + 4) != 1 && !v10 )
      v10 = v8 + 72LL * i;
LABEL_6:
    v14 = v6 + i;
    ++v6;
  }
  if ( !*((_BYTE *)v12 + 4) )
    goto LABEL_6;
  if ( !v10 )
    v10 = v8 + 72LL * i;
  *a3 = v10;
  return 0;
}
