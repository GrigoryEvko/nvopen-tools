// Function: sub_88F240
// Address: 0x88f240
//
__int64 __fastcall sub_88F240(__int64 a1, __int16 a2)
{
  __int64 *v3; // r13
  __int64 *v4; // r14
  __int64 *v5; // rdx
  __int64 v6; // rcx
  char v7; // al
  __int64 result; // rax
  int v9; // esi
  bool v10; // cl
  __int64 v11; // rdx
  bool v12; // zf

  v3 = *(__int64 **)(a1 + 32);
  v4 = *(__int64 **)(a1 + 40);
  v5 = (__int64 *)*v3;
  v6 = *v4;
  v7 = *(_BYTE *)(*v3 + 26);
  if ( *v4 )
  {
    if ( *(_WORD *)(v6 + 24) != a2 )
    {
      if ( v7 != 3 )
        goto LABEL_11;
      result = 0;
      v9 = 1;
      do
LABEL_5:
        v5 = (__int64 *)*v5;
      while ( *((_BYTE *)v5 + 26) == 3 );
      if ( !v9 )
        goto LABEL_7;
LABEL_11:
      result = sub_7AE2C0(a2, *(_DWORD *)(a1 + 24), v5 + 1);
      v10 = a2 == 75;
      *(_QWORD *)result = *v4;
      *v3 = result;
      goto LABEL_8;
    }
    v12 = v7 == 3;
    result = *v4;
    if ( v12 )
    {
      v9 = 0;
      goto LABEL_5;
    }
  }
  else
  {
    v12 = v7 == 3;
    result = 0;
    if ( v12 )
    {
      v9 = 0;
      goto LABEL_5;
    }
  }
LABEL_7:
  *v3 = v6;
  *(_DWORD *)(v6 + 28) = *((_DWORD *)v5 + 7);
  v10 = 0;
LABEL_8:
  *v4 = 0;
  *(_BYTE *)(result + 26) = 5;
  v11 = *(_QWORD *)(a1 + 8);
  *(_BYTE *)(result + 56) = v10;
  *(_QWORD *)(result + 48) = v11;
  *(_QWORD *)(result + 64) = 0;
  return result;
}
