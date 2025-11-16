// Function: sub_3206680
// Address: 0x3206680
//
__int64 __fastcall sub_3206680(_QWORD *a1, unsigned __int64 *a2)
{
  unsigned __int8 v3; // al
  __int64 v4; // rdx
  unsigned __int8 v5; // al
  __int64 v6; // rdi
  __int64 v7; // r14
  __int64 v8; // rdx
  __int64 v9; // rbx
  __int64 result; // rax
  unsigned int v11; // [rsp+8h] [rbp-28h]

  v3 = *((_BYTE *)a2 - 16);
  if ( (v3 & 2) != 0 )
    v4 = *(a2 - 4);
  else
    v4 = (__int64)&a2[-((v3 >> 2) & 0xF) - 2];
  v11 = sub_3206530((__int64)a1, *(unsigned __int8 **)(v4 + 24), 0);
  v5 = *((_BYTE *)a2 - 16);
  if ( (v5 & 2) != 0 )
  {
    v6 = *(_QWORD *)(*(a2 - 4) + 16);
    if ( v6 )
      goto LABEL_5;
LABEL_10:
    sub_31FBCA0(a1, a2);
    return v11;
  }
  v6 = a2[-((v5 >> 2) & 0xF)];
  if ( !v6 )
    goto LABEL_10;
LABEL_5:
  v7 = sub_B91420(v6);
  v9 = v8;
  sub_31FBCA0(a1, a2);
  if ( v11 != 18 )
  {
    if ( v11 == 33 && v9 == 7 && *(_DWORD *)v7 == 1634231159 && *(_WORD *)(v7 + 4) == 24434 )
    {
      result = 113;
      if ( *(_BYTE *)(v7 + 6) == 116 )
        return result;
    }
    return v11;
  }
  if ( v9 != 7 )
    return v11;
  if ( *(_DWORD *)v7 != 1397051976 )
    return v11;
  if ( *(_WORD *)(v7 + 4) != 19541 )
    return v11;
  result = 8;
  if ( *(_BYTE *)(v7 + 6) != 84 )
    return v11;
  return result;
}
