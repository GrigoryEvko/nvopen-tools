// Function: sub_E91F10
// Address: 0xe91f10
//
__int64 __fastcall sub_E91F10(__int64 a1, unsigned int a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  unsigned int v5; // ecx
  int *v6; // rdx
  int v7; // r10d
  int v9; // edx
  int v10; // ebx
  __int64 v11; // [rsp+0h] [rbp-70h]
  __int64 v12; // [rsp+8h] [rbp-68h]
  char v13; // [rsp+20h] [rbp-50h]
  _QWORD v14[4]; // [rsp+30h] [rbp-40h] BYREF
  __int16 v15; // [rsp+50h] [rbp-20h]

  if ( !*(_DWORD *)(a1 + 208) )
    sub_C64ED0("target does not implement codeview register mapping", 1u);
  v3 = *(_QWORD *)(a1 + 200);
  v4 = *(unsigned int *)(a1 + 216);
  if ( !(_DWORD)v4 )
    goto LABEL_8;
  v5 = (v4 - 1) & (37 * a2);
  v6 = (int *)(v3 + 8LL * v5);
  v7 = *v6;
  if ( a2 != *v6 )
  {
    v9 = 1;
    while ( v7 != -1 )
    {
      v10 = v9 + 1;
      v5 = (v4 - 1) & (v9 + v5);
      v6 = (int *)(v3 + 8LL * v5);
      v7 = *v6;
      if ( a2 == *v6 )
        goto LABEL_4;
      v9 = v10;
    }
LABEL_8:
    if ( *(_DWORD *)(a1 + 16) > a2 )
    {
      v13 = 1;
      if ( *(_BYTE *)(*(_QWORD *)(a1 + 72) + *(unsigned int *)(*(_QWORD *)(a1 + 8) + 24LL * a2)) )
      {
        v11 = *(_QWORD *)(a1 + 72) + *(unsigned int *)(*(_QWORD *)(a1 + 8) + 24LL * a2);
        v13 = 3;
      }
    }
    else
    {
      v13 = 9;
      LODWORD(v11) = a2;
    }
    if ( v13 == 1 )
    {
      v15 = 259;
      v14[0] = "unknown codeview register ";
    }
    else
    {
      v14[2] = v11;
      v14[0] = "unknown codeview register ";
      v14[3] = v12;
      LOBYTE(v15) = 3;
      HIBYTE(v15) = v13;
    }
    sub_C64D30((__int64)v14, 1u);
  }
LABEL_4:
  if ( v6 == (int *)(v3 + 8 * v4) )
    goto LABEL_8;
  return (unsigned int)v6[1];
}
