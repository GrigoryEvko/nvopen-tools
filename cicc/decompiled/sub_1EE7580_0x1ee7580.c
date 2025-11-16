// Function: sub_1EE7580
// Address: 0x1ee7580
//
void __fastcall sub_1EE7580(__int64 a1, int *a2, __int64 a3)
{
  int *v3; // rbx
  int *v5; // r12
  int *v6; // r14
  int v7; // r8d
  unsigned int v8; // esi
  __int64 v9; // rcx
  unsigned int v10; // eax
  __int64 v11; // rdi
  _DWORD *v12; // rdx
  int v13; // edx
  int v14; // ecx
  int v15; // r8d
  unsigned int v16; // esi
  __int64 v17; // rcx
  unsigned int v18; // eax
  __int64 v19; // rdi
  _DWORD *v20; // rdx
  int v21; // ecx
  int v22; // edx

  v3 = &a2[2 * a3];
  if ( v3 != a2 )
  {
    v5 = a2;
    v6 = a2;
    do
    {
      v7 = *v6;
      v8 = *v6;
      if ( *v6 < 0 )
        v8 = *(_DWORD *)(a1 + 192) + (v8 & 0x7FFFFFFF);
      v9 = *(unsigned int *)(a1 + 104);
      v10 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 176) + v8);
      if ( v10 >= (unsigned int)v9 )
        goto LABEL_22;
      v11 = *(_QWORD *)(a1 + 96);
      while ( 1 )
      {
        v12 = (_DWORD *)(v11 + 8LL * v10);
        if ( v8 == *v12 )
          break;
        v10 += 256;
        if ( (unsigned int)v9 <= v10 )
          goto LABEL_22;
      }
      if ( v12 == (_DWORD *)(v11 + 8 * v9) )
LABEL_22:
        v13 = 0;
      else
        v13 = v12[1];
      v14 = v6[1];
      v6 += 2;
      sub_1EE5D10(a1, v7, v13, v13 | v14);
    }
    while ( v3 != v6 );
    do
    {
      v15 = *v5;
      v16 = *v5;
      if ( *v5 < 0 )
        v16 = *(_DWORD *)(a1 + 192) + (v16 & 0x7FFFFFFF);
      v17 = *(unsigned int *)(a1 + 104);
      v18 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 176) + v16);
      if ( v18 >= (unsigned int)v17 )
        goto LABEL_23;
      v19 = *(_QWORD *)(a1 + 96);
      while ( 1 )
      {
        v20 = (_DWORD *)(v19 + 8LL * v18);
        if ( v16 == *v20 )
          break;
        v18 += 256;
        if ( (unsigned int)v17 <= v18 )
          goto LABEL_23;
      }
      if ( v20 == (_DWORD *)(v19 + 8 * v17) )
LABEL_23:
        v21 = 0;
      else
        v21 = v20[1];
      v22 = v5[1];
      v5 += 2;
      sub_1EE5E20(a1, v15, v21 | v22, v21);
    }
    while ( v3 != v5 );
  }
}
