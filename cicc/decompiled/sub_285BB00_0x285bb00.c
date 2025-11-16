// Function: sub_285BB00
// Address: 0x285bb00
//
__int64 __fastcall sub_285BB00(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v5; // rdx
  _QWORD *v6; // rax
  __int64 result; // rax
  __int64 v9; // rdx
  int v10; // eax
  __int64 v11; // r15
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  int v15; // eax
  int v16; // eax
  __int64 v17; // rax
  __int64 v18; // rdx
  _QWORD *v19; // rax
  _QWORD *v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rdx
  unsigned int v23; // r15d
  __int64 v24; // rax
  __int64 v25; // r15
  __int64 *v26; // rax
  __int64 v27; // [rsp+8h] [rbp-38h]

  if ( *(_WORD *)(a3 + 24) != 8 )
    goto LABEL_14;
  v5 = *(_QWORD **)(a3 + 48);
  v6 = *(_QWORD **)a1;
  if ( *(_QWORD **)a1 == v5 )
  {
LABEL_21:
    v11 = *(_QWORD *)(a1 + 16);
    sub_D95540(**(_QWORD **)(a3 + 32));
    if ( (unsigned __int8)sub_DFE0F0(v11)
      || (v25 = *(_QWORD *)(a1 + 16), sub_D95540(**(_QWORD **)(a3 + 32)), (unsigned __int8)sub_DFE120(v25)) )
    {
      v15 = *(_DWORD *)(a1 + 56);
      if ( v15 )
      {
        if ( v15 == 1 && !*(_WORD *)(sub_D33D80((_QWORD *)a3, *(_QWORD *)(a1 + 8), v12, v13, v14) + 24) )
        {
          v26 = *(__int64 **)(a3 + 32);
          if ( *(_WORD *)(*v26 + 24) )
          {
            v16 = !sub_DADE90(*(_QWORD *)(a1 + 8), *v26, *(_QWORD *)a1);
            goto LABEL_25;
          }
        }
        goto LABEL_24;
      }
      if ( !*(_BYTE *)(a2 + 16) )
      {
        v21 = sub_D33D80((_QWORD *)a3, *(_QWORD *)(a1 + 8), v12, v13, v14);
        if ( !*(_WORD *)(v21 + 24) )
        {
          v22 = *(_QWORD *)(v21 + 32);
          v23 = *(_DWORD *)(v22 + 32);
          if ( v23 <= 0x40 )
          {
            v24 = *(_QWORD *)(v22 + 24);
LABEL_39:
            v16 = *(_QWORD *)(a2 + 8) != v24;
            goto LABEL_25;
          }
          v27 = *(_QWORD *)(v21 + 32);
          if ( v23 - (unsigned int)sub_C444A0(v22 + 24) <= 0x40 )
          {
            v24 = **(_QWORD **)(v27 + 24);
            goto LABEL_39;
          }
        }
      }
    }
LABEL_24:
    v16 = 1;
LABEL_25:
    *(_DWORD *)(a1 + 32) += v16;
    v17 = *(_QWORD *)(a3 + 32);
    v18 = *(_QWORD *)(v17 + 8);
    if ( *(_QWORD *)(a3 + 40) != 2 || *(_WORD *)(v18 + 24) )
    {
      if ( *(_BYTE *)(a4 + 28) )
      {
        v19 = *(_QWORD **)(a4 + 8);
        v20 = &v19[*(unsigned int *)(a4 + 20)];
        if ( v19 == v20 )
        {
LABEL_44:
          sub_285BB00(a1, a2, v18, a4);
          result = *(unsigned int *)(a1 + 28);
          if ( (_DWORD)result == -1 )
            return result;
          goto LABEL_15;
        }
        while ( *v19 != v18 )
        {
          if ( v20 == ++v19 )
            goto LABEL_44;
        }
      }
      else if ( !sub_C8CA60(a4, *(_QWORD *)(v17 + 8)) )
      {
        v18 = *(_QWORD *)(*(_QWORD *)(a3 + 32) + 8LL);
        goto LABEL_44;
      }
    }
LABEL_14:
    LODWORD(result) = *(_DWORD *)(a1 + 28);
LABEL_15:
    *(_DWORD *)(a1 + 28) = result + 1;
    *(_DWORD *)(a1 + 48) += sub_2850B20(a3, qword_5001228);
    v10 = 0x10000;
    if ( *(_DWORD *)(a1 + 48) <= 0x10000u )
      v10 = *(_DWORD *)(a1 + 48);
    *(_DWORD *)(a1 + 48) = v10;
    result = 0;
    if ( *(_WORD *)(a3 + 24) == 6 )
      result = sub_DAE0A0(*(_QWORD *)(a1 + 8), a3, *(_QWORD *)a1);
    *(_DWORD *)(a1 + 36) += result;
    return result;
  }
  if ( byte_5000B28 )
  {
    while ( v6 )
    {
      v6 = (_QWORD *)*v6;
      if ( v5 == v6 )
        goto LABEL_21;
    }
  }
  result = sub_2851300(a3, *(_QWORD *)(a1 + 8));
  if ( !(_BYTE)result || *(_DWORD *)(a1 + 56) == 1 )
  {
    v9 = *(_QWORD *)(a3 + 48);
    result = *(_QWORD *)a1;
    if ( v9 == *(_QWORD *)a1 )
    {
LABEL_42:
      ++*(_DWORD *)(a1 + 28);
    }
    else
    {
      while ( result )
      {
        result = *(_QWORD *)result;
        if ( v9 == result )
          goto LABEL_42;
      }
      *(_QWORD *)(a1 + 24) = -1;
      *(_QWORD *)(a1 + 32) = -1;
      *(_QWORD *)(a1 + 40) = -1;
      *(_QWORD *)(a1 + 48) = -1;
    }
  }
  return result;
}
