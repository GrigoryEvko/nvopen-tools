// Function: sub_250CC70
// Address: 0x250cc70
//
__int64 __fastcall sub_250CC70(__int64 a1, __int64 *a2)
{
  unsigned __int8 *v3; // r14
  unsigned int v4; // ecx
  char *v5; // rax
  char v6; // al
  unsigned int v7; // r13d
  __int64 v9; // rdi
  unsigned __int8 **v10; // rax
  unsigned __int8 **v11; // rdx

  v3 = sub_250CBE0(a2, (__int64)a2);
  LOBYTE(v4) = (*(_BYTE *)a2 & 3) == 2 || (*a2 & 3) == 3;
  if ( (_BYTE)v4 )
    return 1;
  v5 = (char *)(*a2 & 0xFFFFFFFFFFFFFFFCLL);
  if ( !v5 )
    return 1;
  v6 = *v5;
  v7 = v4;
  if ( v6 )
  {
    if ( v6 != 22 )
      return 1;
  }
  if ( !sub_B2FC80((__int64)v3) && !(unsigned __int8)sub_B2FC00(v3) )
    return 1;
  v9 = *(_QWORD *)(a1 + 208);
  if ( *(_BYTE *)(v9 + 276) )
  {
    v10 = *(unsigned __int8 ***)(v9 + 256);
    v11 = &v10[*(unsigned int *)(v9 + 268)];
    if ( v10 != v11 )
    {
      while ( v3 != *v10 )
      {
        if ( v11 == ++v10 )
          goto LABEL_10;
      }
      return 1;
    }
  }
  else if ( sub_C8CA60(v9 + 248, (__int64)v3) )
  {
    return 1;
  }
LABEL_10:
  if ( !*(_QWORD *)(a1 + 4432) )
    return v7;
  return (*(__int64 (__fastcall **)(__int64, unsigned __int8 *))(a1 + 4440))(a1 + 4416, v3);
}
