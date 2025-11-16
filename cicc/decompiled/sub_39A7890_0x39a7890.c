// Function: sub_39A7890
// Address: 0x39a7890
//
__int64 __fastcall sub_39A7890(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r9
  __int64 v6; // rcx
  __int64 result; // rax
  __int64 v8; // r14
  __int64 v9; // r15
  __int64 v10; // rbx
  __int64 v11; // rdx
  __int64 v12; // rdx
  unsigned __int64 v13; // r8
  __int64 v14; // rdx
  unsigned __int64 v15; // rdx
  int v16; // eax
  unsigned __int64 v17; // rdx
  unsigned int v18; // edi
  __int64 *v19; // rax
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  __int64 v22; // [rsp+8h] [rbp-48h]
  _BYTE v23[52]; // [rsp+1Ch] [rbp-34h] BYREF

  if ( (*(_BYTE *)(a3 + 29) & 8) == 0 )
    goto LABEL_2;
  sub_39A34D0((__int64)a1, a2, 8455);
  v12 = *(unsigned int *)(a3 + 8);
  v13 = *(_QWORD *)(a3 + 32);
  v5 = *(_QWORD *)(a3 + 8 * (3 - v12));
  v14 = *(_QWORD *)(*(_QWORD *)(a3 + 8 * (4 - v12)) - 8LL * *(unsigned int *)(*(_QWORD *)(a3 + 8 * (4 - v12)) + 8LL));
  v15 = *(_QWORD *)(v14 - 8LL * *(unsigned int *)(v14 + 8));
  v16 = *(unsigned __int8 *)v15;
  if ( (_BYTE)v16 != 1 )
  {
    v21 = v15 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (unsigned int)(v16 - 24) >= 2 )
      v21 = 0;
    v18 = *(_DWORD *)(v21 + 32);
    v19 = *(__int64 **)(v21 + 24);
    if ( v18 <= 0x40 )
      goto LABEL_14;
LABEL_20:
    v20 = *v19;
    goto LABEL_15;
  }
  v17 = *(_QWORD *)(v15 + 136) & 0xFFFFFFFFFFFFFFF8LL;
  v18 = *(_DWORD *)(v17 + 32);
  v19 = *(__int64 **)(v17 + 24);
  if ( v18 > 0x40 )
    goto LABEL_20;
LABEL_14:
  v20 = (__int64)((_QWORD)v19 << (64 - (unsigned __int8)v18)) >> (64 - (unsigned __int8)v18);
LABEL_15:
  if ( v13 == *(_QWORD *)(v5 + 32) * (int)v20 )
    goto LABEL_3;
  v23[2] = 0;
  sub_39A3560((__int64)a1, (__int64 *)(a2 + 8), 11, (__int64)v23, v13 >> 3);
LABEL_2:
  v5 = *(_QWORD *)(a3 + 8 * (3LL - *(unsigned int *)(a3 + 8)));
LABEL_3:
  sub_39A6760(a1, a2, v5, 73);
  v6 = sub_39A5D10(a1);
  result = 4LL - *(unsigned int *)(a3 + 8);
  v8 = *(_QWORD *)(a3 + 8 * result);
  if ( v8 )
  {
    result = *(unsigned int *)(v8 + 8);
    if ( (_DWORD)result )
    {
      v9 = *(unsigned int *)(v8 + 8);
      v10 = 0;
      while ( 1 )
      {
        v11 = *(_QWORD *)(v8 + 8 * (v10 - result));
        if ( v11 && *(_WORD *)(v11 + 2) == 33 )
        {
          ++v10;
          v22 = v6;
          result = sub_39A5B50((__int64)a1, a2, v11, v6);
          v6 = v22;
          if ( v10 == v9 )
            return result;
        }
        else if ( ++v10 == v9 )
        {
          return result;
        }
        result = *(unsigned int *)(v8 + 8);
      }
    }
  }
  return result;
}
