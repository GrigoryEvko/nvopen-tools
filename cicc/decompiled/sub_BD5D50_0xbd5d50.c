// Function: sub_BD5D50
// Address: 0xbd5d50
//
__int64 __fastcall sub_BD5D50(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r13
  __int64 v3; // rax
  unsigned int v4; // r12d
  __int64 *v5; // rax
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rdx
  _QWORD *v12; // rbx
  __int64 *v13; // rax

  v1 = *(_QWORD *)(a1 + 8);
  **(_QWORD **)(a1 + 16) = v1;
  if ( v1 )
    *(_QWORD *)(v1 + 16) = *(_QWORD *)(a1 + 16);
  v2 = *(_QWORD *)(a1 + 24);
  if ( *(_BYTE *)v2 != 85
    || (v3 = *(_QWORD *)(v2 - 32)) == 0
    || *(_BYTE *)v3
    || *(_QWORD *)(v3 + 24) != *(_QWORD *)(v2 + 80)
    || (*(_BYTE *)(v3 + 33) & 0x20) == 0
    || *(_DWORD *)(v3 + 36) != 11 )
  {
    BUG();
  }
  v4 = sub_BD2910(a1);
  if ( v4 )
  {
    v9 = sub_ACA8A0(*(__int64 ***)(*(_QWORD *)a1 + 8LL));
    if ( *(_QWORD *)a1 )
    {
      v10 = *(_QWORD *)(a1 + 8);
      **(_QWORD **)(a1 + 16) = v10;
      if ( v10 )
        *(_QWORD *)(v10 + 16) = *(_QWORD *)(a1 + 16);
    }
    *(_QWORD *)a1 = v9;
    if ( v9 )
    {
      v11 = *(_QWORD *)(v9 + 16);
      *(_QWORD *)(a1 + 8) = v11;
      if ( v11 )
        *(_QWORD *)(v11 + 16) = a1 + 8;
      *(_QWORD *)(a1 + 16) = v9 + 16;
      *(_QWORD *)(v9 + 16) = a1;
    }
    v12 = (_QWORD *)sub_B49810(v2, v4);
    v13 = (__int64 *)sub_BD5C60(v2);
    result = sub_B71A20(*v13, "ignore", 6u);
    *v12 = result;
  }
  else
  {
    v5 = (__int64 *)sub_BD5C60(v2);
    result = sub_ACD6D0(v5);
    if ( *(_QWORD *)a1 )
    {
      v7 = *(_QWORD *)(a1 + 8);
      **(_QWORD **)(a1 + 16) = v7;
      if ( v7 )
        *(_QWORD *)(v7 + 16) = *(_QWORD *)(a1 + 16);
    }
    *(_QWORD *)a1 = result;
    if ( result )
    {
      v8 = *(_QWORD *)(result + 16);
      *(_QWORD *)(a1 + 8) = v8;
      if ( v8 )
        *(_QWORD *)(v8 + 16) = a1 + 8;
      *(_QWORD *)(a1 + 16) = result + 16;
      *(_QWORD *)(result + 16) = a1;
    }
  }
  return result;
}
