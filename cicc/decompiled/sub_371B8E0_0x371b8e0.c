// Function: sub_371B8E0
// Address: 0x371b8e0
//
unsigned __int64 __fastcall sub_371B8E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  int v9; // edx
  int v10; // esi
  __int64 v11; // r8
  __int64 v12; // rdx
  unsigned int v13; // ecx
  __int64 *v14; // rdi
  __int64 v15; // r9
  __int64 v16; // rdx
  bool v17; // of
  unsigned __int64 v18; // rdx
  unsigned __int64 result; // rax
  int v20; // edi
  unsigned __int64 v21; // rdx
  __int64 v22; // rdx
  int v23; // r11d
  bool v24; // cc

  v8 = sub_371B7D0(a1, a2, a3, a4, a5, a6);
  v10 = v9;
  v11 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
  v12 = *(unsigned int *)(*(_QWORD *)a1 + 24LL);
  if ( (_DWORD)v12 )
  {
    v13 = (v12 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v14 = (__int64 *)(v11 + 8LL * v13);
    v15 = *v14;
    if ( a2 == *v14 )
    {
LABEL_3:
      if ( v14 != (__int64 *)(v11 + 8 * v12) )
      {
        if ( v10 == 1 )
        {
          v22 = *(_QWORD *)(a1 + 16);
          *(_DWORD *)(a1 + 24) = 1;
          v17 = __OFSUB__(v22, v8);
          v18 = v22 - v8;
          if ( !v17 )
            goto LABEL_6;
        }
        else
        {
          v16 = *(_QWORD *)(a1 + 16);
          v17 = __OFSUB__(v16, v8);
          v18 = v16 - v8;
          if ( !v17 )
          {
LABEL_6:
            result = v18;
LABEL_7:
            *(_QWORD *)(a1 + 16) = result;
            return result;
          }
        }
        v24 = v8 <= 0;
        result = 0x8000000000000000LL;
        if ( v24 )
          result = 0x7FFFFFFFFFFFFFFFLL;
        goto LABEL_7;
      }
    }
    else
    {
      v20 = 1;
      while ( v15 != -4096 )
      {
        v23 = v20 + 1;
        v13 = (v12 - 1) & (v20 + v13);
        v14 = (__int64 *)(v11 + 8LL * v13);
        v15 = *v14;
        if ( a2 == *v14 )
          goto LABEL_3;
        v20 = v23;
      }
    }
  }
  if ( v10 == 1 )
  {
    v17 = __OFADD__(*(_QWORD *)(a1 + 32), v8);
    v21 = *(_QWORD *)(a1 + 32) + v8;
    *(_DWORD *)(a1 + 40) = 1;
    if ( !v17 )
      goto LABEL_12;
LABEL_20:
    v24 = v8 <= 0;
    result = 0x7FFFFFFFFFFFFFFFLL;
    if ( v24 )
      result = 0x8000000000000000LL;
    goto LABEL_13;
  }
  v21 = *(_QWORD *)(a1 + 32) + v8;
  if ( __OFADD__(*(_QWORD *)(a1 + 32), v8) )
    goto LABEL_20;
LABEL_12:
  result = v21;
LABEL_13:
  *(_QWORD *)(a1 + 32) = result;
  return result;
}
