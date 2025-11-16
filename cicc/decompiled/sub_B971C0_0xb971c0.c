// Function: sub_B971C0
// Address: 0xb971c0
//
void __fastcall sub_B971C0(__int64 a1, __int64 a2, char a3, char a4, __int64 *a5, __int64 a6, __int64 *a7, __int64 a8)
{
  __int64 *v8; // r14
  __int64 *v9; // r13
  __int64 *v10; // r12
  unsigned int v11; // ebx
  __int64 v12; // rdx
  unsigned int v13; // esi
  __int64 *v14; // r12
  __int64 v15; // rdx
  unsigned int v16; // esi

  v8 = &a5[a6];
  *(_BYTE *)a1 = a3;
  *(_BYTE *)(a1 + 1) = a4 & 0x7F;
  v9 = a7;
  *(_WORD *)(a1 + 2) = 0;
  *(_DWORD *)(a1 + 4) = 0;
  *(_QWORD *)(a1 + 8) = a2 & 0xFFFFFFFFFFFFFFFBLL;
  if ( a5 == v8 )
  {
    v11 = 0;
  }
  else
  {
    v10 = a5;
    v11 = 0;
    do
    {
      v12 = *v10;
      v13 = v11;
      ++v10;
      ++v11;
      sub_B97110(a1, v13, v12);
    }
    while ( v8 != v10 );
  }
  v14 = &a7[a8];
  if ( a7 != v14 )
  {
    do
    {
      v15 = *v9;
      v16 = v11;
      ++v9;
      ++v11;
      sub_B97110(a1, v16, v15);
    }
    while ( v14 != v9 );
  }
  if ( (*(_BYTE *)(a1 + 1) & 0x7F) == 0 )
    sub_B91600(a1);
}
