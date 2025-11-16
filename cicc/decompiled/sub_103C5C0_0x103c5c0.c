// Function: sub_103C5C0
// Address: 0x103c5c0
//
__int64 __fastcall sub_103C5C0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // r15
  __int64 v5; // r14
  __int64 v6; // rbx
  unsigned int v7; // r13d
  __int64 v8; // rdi
  unsigned int v9; // edx
  unsigned int v10; // esi
  __int64 *v11; // rdx
  __int64 v12; // r8
  __int64 v13; // rsi
  __int64 v14; // rbx
  unsigned int v15; // edx
  int v16; // edx
  unsigned int v17; // esi
  __int64 v18; // rdx
  __int64 v19; // rsi
  __int64 v20; // rsi
  unsigned int v21; // edx
  __int64 v22; // rsi
  unsigned __int64 *v23; // r13
  unsigned __int64 *v24; // rbx
  unsigned __int64 v25; // rdx
  __int64 v26; // rdi
  unsigned __int64 *v27; // rdi
  int v28; // eax
  __int64 *v29; // rdi
  __int64 v30; // rdx
  __int64 v31; // rdx
  int v32; // edx
  int v33; // r9d
  unsigned int v34; // esi
  unsigned int v35; // edi
  __int64 v36; // [rsp+8h] [rbp-48h]
  int v38; // [rsp+18h] [rbp-38h]

  result = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( result == a2 + 48 )
    goto LABEL_52;
  if ( !result )
    BUG();
  v4 = result - 24;
  result = (unsigned int)*(unsigned __int8 *)(result - 24) - 30;
  if ( (unsigned int)result > 0xA )
  {
LABEL_52:
    v5 = *(_QWORD *)(a1 + 72);
    v6 = *(unsigned int *)(a1 + 88);
  }
  else
  {
    result = sub_B46E30(v4);
    v5 = *(_QWORD *)(a1 + 72);
    v6 = *(unsigned int *)(a1 + 88);
    v38 = result;
    if ( (_DWORD)result )
    {
      v7 = 0;
      while ( 1 )
      {
        result = sub_B46EC0(v4, v7);
        v13 = *(_QWORD *)(a1 + 8);
        if ( result )
        {
          v8 = (unsigned int)(*(_DWORD *)(result + 44) + 1);
          v9 = *(_DWORD *)(result + 44) + 1;
        }
        else
        {
          v8 = 0;
          v9 = 0;
        }
        if ( v9 >= *(_DWORD *)(v13 + 32) || !*(_QWORD *)(*(_QWORD *)(v13 + 24) + 8 * v8) || !(_DWORD)v6 )
          goto LABEL_14;
        v10 = (v6 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
        v11 = (__int64 *)(v5 + 16LL * v10);
        v12 = *v11;
        if ( result != *v11 )
        {
          v32 = 1;
          while ( v12 != -4096 )
          {
            v33 = v32 + 1;
            v10 = (v6 - 1) & (v32 + v10);
            v11 = (__int64 *)(v5 + 16LL * v10);
            v12 = *v11;
            if ( result == *v11 )
              goto LABEL_11;
            v32 = v33;
          }
          goto LABEL_14;
        }
LABEL_11:
        result = v5 + 16LL * (unsigned int)v6;
        if ( v11 == (__int64 *)result )
          goto LABEL_14;
        result = *(_QWORD *)(v11[1] + 8);
        if ( !result )
          BUG();
        if ( *(_BYTE *)(result - 32) == 28 )
        {
          v14 = *(_QWORD *)(a1 + 128);
          v15 = *(_DWORD *)(result - 28) & 0x7FFFFFF;
          if ( v15 == *(_DWORD *)(result + 44) )
          {
            v36 = result;
            v34 = v15 + (v15 >> 1);
            if ( v34 < 2 )
              v34 = 2;
            *(_DWORD *)(result + 44) = v34;
            sub_BD2A80(result - 32, v34, 1);
            result = v36;
            v15 = *(_DWORD *)(v36 - 28) & 0x7FFFFFF;
          }
          v16 = (v15 + 1) & 0x7FFFFFF;
          v17 = v16 | *(_DWORD *)(result - 28) & 0xF8000000;
          v18 = *(_QWORD *)(result - 40) + 32LL * (unsigned int)(v16 - 1);
          *(_DWORD *)(result - 28) = v17;
          if ( *(_QWORD *)v18 )
          {
            v19 = *(_QWORD *)(v18 + 8);
            **(_QWORD **)(v18 + 16) = v19;
            if ( v19 )
              *(_QWORD *)(v19 + 16) = *(_QWORD *)(v18 + 16);
          }
          *(_QWORD *)v18 = v14;
          if ( v14 )
          {
            v20 = *(_QWORD *)(v14 + 16);
            *(_QWORD *)(v18 + 8) = v20;
            if ( v20 )
              *(_QWORD *)(v20 + 16) = v18 + 8;
            *(_QWORD *)(v18 + 16) = v14 + 16;
            *(_QWORD *)(v14 + 16) = v18;
          }
          ++v7;
          *(_QWORD *)(*(_QWORD *)(result - 40)
                    + 32LL * *(unsigned int *)(result + 44)
                    + 8LL * ((*(_DWORD *)(result - 28) & 0x7FFFFFFu) - 1)) = a2;
          v5 = *(_QWORD *)(a1 + 72);
          v6 = *(unsigned int *)(a1 + 88);
          if ( v38 == v7 )
            break;
        }
        else
        {
LABEL_14:
          if ( v38 == ++v7 )
            break;
        }
      }
    }
  }
  if ( (_DWORD)v6 )
  {
    v21 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    result = v5 + 16LL * v21;
    v22 = *(_QWORD *)result;
    if ( a2 == *(_QWORD *)result )
    {
LABEL_28:
      if ( result == 16 * v6 + v5 )
        return result;
      v23 = *(unsigned __int64 **)(result + 8);
      v24 = (unsigned __int64 *)v23[1];
      if ( v23 == v24 )
        return result;
      while ( 1 )
      {
        while ( 1 )
        {
          v27 = v24;
          v24 = (unsigned __int64 *)v24[1];
          v28 = *((unsigned __int8 *)v27 - 32);
          if ( v28 != 26 )
            break;
          result = *(_QWORD *)(a1 + 128);
          v29 = (__int64 *)(v27 - 8);
LABEL_36:
          if ( *v29 )
          {
            v30 = v29[1];
            *(_QWORD *)v29[2] = v30;
            if ( v30 )
              *(_QWORD *)(v30 + 16) = v29[2];
          }
          *v29 = result;
          if ( !result )
            goto LABEL_33;
          v31 = *(_QWORD *)(result + 16);
          v29[1] = v31;
          if ( v31 )
          {
            v22 = (__int64)(v29 + 1);
            *(_QWORD *)(v31 + 16) = v29 + 1;
          }
          v29[2] = result + 16;
          *(_QWORD *)(result + 16) = v29;
          if ( v23 == v24 )
            return result;
        }
        if ( v28 == 27 )
        {
          result = *(_QWORD *)(a1 + 128);
          v29 = (__int64 *)(v27 - 12);
          goto LABEL_36;
        }
        v25 = *v27 & 0xFFFFFFFFFFFFFFF8LL;
        *v24 = v25 | *v24 & 7;
        *(_QWORD *)(v25 + 8) = v24;
        *v27 &= 7u;
        v26 = (__int64)(v27 - 4);
        *(_QWORD *)(v26 + 40) = 0;
        result = sub_BD72D0(v26, v22);
LABEL_33:
        if ( v23 == v24 )
          return result;
      }
    }
    result = 1;
    while ( v22 != -4096 )
    {
      v35 = result + 1;
      v21 = (v6 - 1) & (result + v21);
      result = v5 + 16LL * v21;
      v22 = *(_QWORD *)result;
      if ( a2 == *(_QWORD *)result )
        goto LABEL_28;
      result = v35;
    }
  }
  return result;
}
