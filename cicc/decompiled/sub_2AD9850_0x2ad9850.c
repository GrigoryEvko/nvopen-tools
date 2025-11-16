// Function: sub_2AD9850
// Address: 0x2ad9850
//
__int64 __fastcall sub_2AD9850(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rsi
  __int64 result; // rax
  _QWORD *v10; // rdi
  __int64 v11; // rcx
  int v12; // ecx
  __int64 v13; // rdx
  int v14; // edx
  __int64 v15; // rbx
  __int64 v16; // r9
  unsigned int v17; // esi
  int v18; // eax
  __int64 v19; // rdx
  int v20; // eax
  __int64 v21; // r8
  __int64 v22; // rbx
  __int64 v23; // [rsp+0h] [rbp-20h] BYREF
  _QWORD v24[3]; // [rsp+8h] [rbp-18h] BYREF

  if ( !*(_DWORD *)(a1 + 16) )
  {
    v8 = *(unsigned int *)(a1 + 40);
    result = *(_QWORD *)(a1 + 32);
    v10 = (_QWORD *)(result + 8 * v8);
    v11 = (8 * v8) >> 3;
    if ( (8 * v8) >> 5 )
    {
      v12 = *(_DWORD *)a2;
      v13 = result + 32 * ((8 * v8) >> 5);
      while ( 1 )
      {
        if ( *(_DWORD *)result == v12 )
        {
          a6 = *((unsigned __int8 *)a2 + 4);
          if ( *(_BYTE *)(result + 4) == (_BYTE)a6 )
            break;
        }
        if ( v12 == *(_DWORD *)(result + 8) && *(_BYTE *)(result + 12) == *((_BYTE *)a2 + 4) )
        {
          result += 8;
          if ( v10 != (_QWORD *)result )
            return result;
          goto LABEL_16;
        }
        if ( v12 == *(_DWORD *)(result + 16) && *(_BYTE *)(result + 20) == *((_BYTE *)a2 + 4) )
        {
          result += 16;
          if ( v10 != (_QWORD *)result )
            return result;
          goto LABEL_16;
        }
        if ( v12 == *(_DWORD *)(result + 24) )
        {
          a5 = *((unsigned __int8 *)a2 + 4);
          if ( *(_BYTE *)(result + 28) == (_BYTE)a5 )
          {
            result += 24;
            if ( v10 != (_QWORD *)result )
              return result;
            goto LABEL_16;
          }
        }
        result += 32;
        if ( v13 == result )
        {
          v11 = ((__int64)v10 - result) >> 3;
          goto LABEL_10;
        }
      }
LABEL_21:
      if ( v10 != (_QWORD *)result )
        return result;
      goto LABEL_16;
    }
LABEL_10:
    switch ( v11 )
    {
      case 2LL:
        v14 = *(_DWORD *)a2;
        break;
      case 3LL:
        v14 = *(_DWORD *)a2;
        if ( *(_DWORD *)result == *(_DWORD *)a2 && *(_BYTE *)(result + 4) == *((_BYTE *)a2 + 4) )
          goto LABEL_21;
        result += 8;
        break;
      case 1LL:
        v14 = *(_DWORD *)a2;
LABEL_14:
        if ( *(_DWORD *)result == v14 && *(_BYTE *)(result + 4) == *((_BYTE *)a2 + 4) )
          goto LABEL_21;
LABEL_16:
        v15 = *a2;
        if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
        {
          sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v8 + 1, 8u, a5, a6);
          v10 = (_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL * *(unsigned int *)(a1 + 40));
        }
        *v10 = v15;
        result = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
        *(_DWORD *)(a1 + 40) = result;
        if ( (unsigned int)result > 2 )
          return sub_2AD96C0(a1);
        return result;
      default:
        goto LABEL_16;
    }
    if ( *(_DWORD *)result == v14 && *(_BYTE *)(result + 4) == *((_BYTE *)a2 + 4) )
      goto LABEL_21;
    result += 8;
    goto LABEL_14;
  }
  result = sub_2AC3BB0(a1, (int *)a2, &v23);
  if ( (_BYTE)result )
    return result;
  v17 = *(_DWORD *)(a1 + 24);
  v18 = *(_DWORD *)(a1 + 16);
  v19 = v23;
  ++*(_QWORD *)a1;
  v20 = v18 + 1;
  v21 = 2 * v17;
  v24[0] = v19;
  if ( 4 * v20 >= 3 * v17 )
  {
    v17 *= 2;
  }
  else if ( v17 - *(_DWORD *)(a1 + 20) - v20 > v17 >> 3 )
  {
    goto LABEL_35;
  }
  sub_2AD9490(a1, v17);
  sub_2AC3BB0(a1, (int *)a2, v24);
  v19 = v24[0];
  v20 = *(_DWORD *)(a1 + 16) + 1;
LABEL_35:
  *(_DWORD *)(a1 + 16) = v20;
  if ( *(_DWORD *)v19 != -1 || !*(_BYTE *)(v19 + 4) )
    --*(_DWORD *)(a1 + 20);
  *(_DWORD *)v19 = *(_DWORD *)a2;
  *(_BYTE *)(v19 + 4) = *((_BYTE *)a2 + 4);
  result = *(unsigned int *)(a1 + 40);
  v22 = *a2;
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), result + 1, 8u, v21, v16);
    result = *(unsigned int *)(a1 + 40);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * result) = v22;
  ++*(_DWORD *)(a1 + 40);
  return result;
}
