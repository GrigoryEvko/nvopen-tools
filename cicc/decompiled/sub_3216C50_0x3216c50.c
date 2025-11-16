// Function: sub_3216C50
// Address: 0x3216c50
//
__int64 __fastcall sub_3216C50(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, unsigned __int64 a5, __int64 a6)
{
  int v6; // r15d
  unsigned int v8; // r15d
  __int64 *v9; // rax
  __int64 v10; // rax
  unsigned __int64 v11; // rax
  __int64 *v12; // rbx
  __int64 *v13; // rax
  __int64 v14; // rax
  unsigned __int64 v15; // rax
  __int64 *v16; // rbx

  v6 = a4;
  sub_3216910(a3, a1, (__int64)a3, a4, a5, a6);
  *(_DWORD *)(a1 + 16) = v6;
  v8 = sub_F03EF0(*(unsigned int *)(a1 + 24)) + v6;
  v9 = *(__int64 **)(a1 + 8);
  if ( v9 )
  {
    v10 = *v9;
    do
    {
      v11 = v10 & 0xFFFFFFFFFFFFFFF8LL;
      v12 = (__int64 *)v11;
      if ( !v11 )
        break;
      v8 += sub_3215DE0(v11 + 8, a2);
      v10 = *v12;
    }
    while ( (*v12 & 4) == 0 );
  }
  v13 = *(__int64 **)(a1 + 32);
  if ( *(_BYTE *)(a1 + 30) )
  {
    if ( !v13 )
    {
LABEL_10:
      ++v8;
      goto LABEL_11;
    }
LABEL_7:
    v14 = *v13;
    do
    {
      v15 = v14 & 0xFFFFFFFFFFFFFFF8LL;
      v16 = (__int64 *)v15;
      if ( !v15 )
        break;
      v8 = sub_3216C50(v15, a2, a3, v8);
      v14 = *v16;
    }
    while ( (*v16 & 4) == 0 );
    goto LABEL_10;
  }
  if ( v13 )
    goto LABEL_7;
LABEL_11:
  *(_DWORD *)(a1 + 20) = v8 - *(_DWORD *)(a1 + 16);
  return v8;
}
