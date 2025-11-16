// Function: sub_26C9BA0
// Address: 0x26c9ba0
//
__int64 __fastcall sub_26C9BA0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // r13
  int v5; // r14d
  unsigned int v6; // r15d
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rbx
  unsigned __int8 v10; // dl
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rax
  _QWORD *v14; // rcx
  unsigned int v15; // esi
  __int64 v16; // r9
  __int64 *v17; // r8
  __int64 v18; // rdi
  __int64 v19; // rax
  unsigned __int8 v20; // dl
  __int64 v21; // rcx
  int v22; // r11d
  __int64 *v23; // rdx
  int v24; // eax
  int v25; // edi
  __int64 v26; // r8
  _QWORD *v27; // [rsp+8h] [rbp-48h]
  __int64 v28; // [rsp+10h] [rbp-40h] BYREF
  __int64 *v29; // [rsp+18h] [rbp-38h] BYREF

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  result = sub_BA8DC0(a2, (__int64)"llvm.pseudo_probe_desc", 22);
  if ( result )
  {
    v4 = result;
    result = sub_B91A00(result);
    v5 = result;
    if ( (_DWORD)result )
    {
      v6 = 0;
      while ( 1 )
      {
        v19 = sub_B91A10(v4, v6);
        v20 = *(_BYTE *)(v19 - 16);
        v21 = v19 - 16;
        if ( (v20 & 2) != 0 )
          v7 = *(_QWORD *)(v19 - 32);
        else
          v7 = v21 - 8LL * ((v20 >> 2) & 0xF);
        if ( **(_BYTE **)v7 != 1 )
          goto LABEL_40;
        v8 = *(_QWORD *)(*(_QWORD *)v7 + 136LL);
        if ( *(_BYTE *)v8 != 17 )
          goto LABEL_40;
        v9 = *(_QWORD *)(v8 + 24);
        if ( *(_DWORD *)(v8 + 32) > 0x40u )
          v9 = *(_QWORD *)v9;
        v28 = v9;
        v10 = *(_BYTE *)(v19 - 16);
        v11 = (v10 & 2) != 0 ? *(_QWORD *)(v19 - 32) : v21 - 8LL * ((v10 >> 2) & 0xF);
        v12 = *(_QWORD *)(v11 + 8);
        if ( *(_BYTE *)v12 != 1 || (v13 = *(_QWORD *)(v12 + 136), *(_BYTE *)v13 != 17) )
LABEL_40:
          BUG();
        v14 = *(_QWORD **)(v13 + 24);
        if ( *(_DWORD *)(v13 + 32) > 0x40u )
          v14 = (_QWORD *)*v14;
        v15 = *(_DWORD *)(a1 + 24);
        if ( !v15 )
          break;
        v16 = *(_QWORD *)(a1 + 8);
        result = (v15 - 1) & ((unsigned int)((0xBF58476D1CE4E5B9LL * v9) >> 31) ^ (484763065 * (_DWORD)v9));
        v17 = (__int64 *)(v16 + 24 * result);
        v18 = *v17;
        if ( v9 != *v17 )
        {
          v22 = 1;
          v23 = 0;
          while ( v18 != -1 )
          {
            if ( v23 || v18 != -2 )
              v17 = v23;
            result = (v15 - 1) & (v22 + (_DWORD)result);
            v18 = *(_QWORD *)(v16 + 24LL * (unsigned int)result);
            if ( v9 == v18 )
              goto LABEL_17;
            ++v22;
            v23 = v17;
            v17 = (__int64 *)(v16 + 24LL * (unsigned int)result);
          }
          v24 = *(_DWORD *)(a1 + 16);
          if ( !v23 )
            v23 = v17;
          ++*(_QWORD *)a1;
          v25 = v24 + 1;
          v29 = v23;
          if ( 4 * (v24 + 1) < 3 * v15 )
          {
            v26 = v9;
            result = v15 - *(_DWORD *)(a1 + 20) - v25;
            if ( (unsigned int)result > v15 >> 3 )
            {
LABEL_28:
              *(_DWORD *)(a1 + 16) = v25;
              if ( *v23 != -1 )
                --*(_DWORD *)(a1 + 20);
              *v23 = v26;
              v23[1] = v9;
              v23[2] = (__int64)v14;
              goto LABEL_17;
            }
            v27 = v14;
LABEL_33:
            sub_26C9990(a1, v15);
            sub_26C2D70(a1, &v28, &v29);
            result = *(unsigned int *)(a1 + 16);
            v26 = v28;
            v23 = v29;
            v14 = v27;
            v25 = result + 1;
            goto LABEL_28;
          }
LABEL_32:
          v27 = v14;
          v15 *= 2;
          goto LABEL_33;
        }
LABEL_17:
        if ( v5 == ++v6 )
          return result;
      }
      ++*(_QWORD *)a1;
      v29 = 0;
      goto LABEL_32;
    }
  }
  return result;
}
