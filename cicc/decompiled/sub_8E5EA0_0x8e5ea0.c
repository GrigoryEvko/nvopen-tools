// Function: sub_8E5EA0
// Address: 0x8e5ea0
//
unsigned __int8 *__fastcall sub_8E5EA0(unsigned __int8 *a1, __int64 a2)
{
  unsigned __int8 *v2; // r12
  unsigned __int8 v3; // bl
  __int64 v5; // rax
  __int64 v6; // [rsp+8h] [rbp-78h] BYREF
  char s[112]; // [rsp+10h] [rbp-70h] BYREF

  v2 = a1;
  v3 = *a1;
  if ( *a1 != 118 && v3 != 104 )
  {
    if ( !*(_DWORD *)(a2 + 24) )
    {
      ++*(_QWORD *)(a2 + 32);
      ++*(_QWORD *)(a2 + 48);
      *(_DWORD *)(a2 + 24) = 1;
    }
    return v2;
  }
  if ( !*(_QWORD *)(a2 + 32) )
    sub_8E5790("(offset ", a2);
  v2 = sub_8E5810(a1 + 1, &v6, a2);
  sprintf(s, "%ld", v6);
  v5 = *(_QWORD *)(a2 + 32);
  if ( v5 )
  {
    if ( v3 != 118 )
      goto LABEL_10;
LABEL_16:
    if ( *v2 == 95 )
    {
      ++v2;
    }
    else if ( !*(_DWORD *)(a2 + 24) )
    {
      ++*(_QWORD *)(a2 + 32);
      ++*(_QWORD *)(a2 + 48);
      *(_DWORD *)(a2 + 24) = 1;
    }
    v2 = sub_8E5810(v2, &v6, a2);
    sprintf(s, "%ld", v6);
    v5 = *(_QWORD *)(a2 + 32);
    if ( !v5 )
    {
      sub_8E5790((unsigned __int8 *)s, a2);
      v5 = *(_QWORD *)(a2 + 32);
    }
    goto LABEL_10;
  }
  sub_8E5790((unsigned __int8 *)s, a2);
  v5 = *(_QWORD *)(a2 + 32);
  if ( v3 == 118 )
  {
    if ( !v5 )
      sub_8E5790(", virtual offset ", a2);
    goto LABEL_16;
  }
LABEL_10:
  if ( *v2 == 95 )
  {
    ++v2;
  }
  else if ( !*(_DWORD *)(a2 + 24) )
  {
    ++v5;
    ++*(_QWORD *)(a2 + 48);
    *(_DWORD *)(a2 + 24) = 1;
    *(_QWORD *)(a2 + 32) = v5;
  }
  if ( !v5 )
    sub_8E5790((unsigned __int8 *)") ", a2);
  return v2;
}
