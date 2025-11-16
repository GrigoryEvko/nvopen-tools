// Function: sub_8EFAB0
// Address: 0x8efab0
//
void __fastcall sub_8EFAB0(__int64 a1, __int64 a2)
{
  unsigned int v3; // ecx
  __int64 v4; // rsi
  __int64 v5; // rdx
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rax
  __int64 v8; // rdx
  unsigned __int64 v9; // rax

  v3 = *(_DWORD *)(a2 + 2088);
  v4 = *(unsigned int *)(a1 + 2088);
  if ( (unsigned int)v4 < v3 )
  {
    memset((void *)(a1 + 4 * v4 + 8), 0, 4LL * (v3 - (unsigned int)v4));
    v3 = *(_DWORD *)(a2 + 2088);
    *(_DWORD *)(a1 + 2088) = v3;
    LODWORD(v4) = v3;
    if ( v3 )
      goto LABEL_3;
  }
  else if ( v3 )
  {
LABEL_3:
    v5 = 1;
    v6 = 0;
    do
    {
      v7 = *(unsigned int *)(a2 + 4 * v5 + 4) + (unsigned __int64)*(unsigned int *)(a1 + 4 * v5 + 4) + v6;
      v3 = v5;
      *(_DWORD *)(a1 + 4 * v5++ + 4) = v7;
      v6 = HIDWORD(v7);
    }
    while ( *(_DWORD *)(a2 + 2088) > (unsigned int)(v5 - 1) );
    if ( v3 >= (unsigned int)v4 )
      goto LABEL_8;
    goto LABEL_6;
  }
  if ( !(_DWORD)v4 )
    return;
  v6 = 0;
LABEL_6:
  v8 = v3;
  do
  {
    v9 = *(unsigned int *)(a1 + 4 * v8 + 8) + v6;
    *(_DWORD *)(a1 + 4 * v8++ + 8) = v9;
    v6 = HIDWORD(v9);
  }
  while ( (unsigned int)v4 > (unsigned int)v8 );
LABEL_8:
  if ( v6 )
  {
    *(_DWORD *)(a1 + 4LL * (unsigned int)v4 + 8) = 1;
    *(_DWORD *)(a1 + 2088) = v4 + 1;
  }
}
