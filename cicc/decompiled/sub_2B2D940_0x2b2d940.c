// Function: sub_2B2D940
// Address: 0x2b2d940
//
__int64 __fastcall sub_2B2D940(__int64 a1, char a2, __int64 a3, _QWORD *a4)
{
  int v4; // eax
  __int64 *v5; // rcx
  int v6; // edx
  __int64 result; // rax
  __int64 v8; // r8
  __int64 v9; // rdx
  unsigned int v10; // eax
  __int64 v11[3]; // [rsp+8h] [rbp-58h] BYREF
  char v12; // [rsp+20h] [rbp-40h] BYREF

  v4 = *(_DWORD *)(a1 + 8);
  v11[0] = a1;
  if ( v4 == 1 )
  {
    v8 = **(_QWORD **)a1;
    if ( (*(_DWORD *)(v8 + 104) & 0xFFFFFFFD) == 0 )
      return 1;
    if ( !a2 )
      return 0;
    if ( (unsigned __int8)sub_2B266F0(v11, (_QWORD **)v8, *(unsigned int *)(v8 + 8), a4) )
    {
      v9 = **(_QWORD **)a1;
      v10 = *(_DWORD *)(v9 + 120);
      if ( !v10 )
        v10 = *(_DWORD *)(v9 + 8);
      if ( v10 > 2 )
        return 1;
    }
    v4 = *(_DWORD *)(a1 + 8);
  }
  if ( v4 != 2 )
    return 0;
  v5 = *(__int64 **)a1;
  v11[1] = (__int64)&v12;
  v11[2] = 0xC00000000LL;
  v6 = *(_DWORD *)(*v5 + 104);
  if ( !v6 )
  {
    if ( !(unsigned __int8)sub_2B266F0(v11, (_QWORD **)v5[1], *(unsigned int *)(*v5 + 8), v5) )
    {
      v5 = *(__int64 **)a1;
      v6 = *(_DWORD *)(**(_QWORD **)a1 + 104LL);
      goto LABEL_4;
    }
    return 1;
  }
LABEL_4:
  result = 0;
  if ( v6 != 3 && (*(_DWORD *)(v5[1] + 104) != 3 || (unsigned int)(v6 - 1) <= 1) )
    return 1;
  return result;
}
