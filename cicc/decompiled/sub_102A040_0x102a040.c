// Function: sub_102A040
// Address: 0x102a040
//
unsigned __int64 __fastcall sub_102A040(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rbx
  unsigned __int64 result; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  unsigned int v8; // eax
  __int64 *v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r14
  __int64 *v12; // rax
  __int64 v13; // rsi
  __int64 *v14; // r15
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // rdi
  __int64 v18; // r14
  __int64 v19; // r15
  char v20; // al
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  unsigned __int64 v29; // [rsp+10h] [rbp-280h]
  __int64 *v30; // [rsp+18h] [rbp-278h]
  __int64 *v31; // [rsp+20h] [rbp-270h] BYREF
  __int64 v32; // [rsp+28h] [rbp-268h]
  _QWORD v33[16]; // [rsp+30h] [rbp-260h] BYREF
  __int64 *v34; // [rsp+B0h] [rbp-1E0h] BYREF
  __int64 v35; // [rsp+B8h] [rbp-1D8h]
  _BYTE v36[128]; // [rsp+C0h] [rbp-1D0h] BYREF
  __int64 v37; // [rsp+140h] [rbp-150h] BYREF
  __int64 *v38; // [rsp+148h] [rbp-148h]
  __int64 v39; // [rsp+150h] [rbp-140h]
  int v40; // [rsp+158h] [rbp-138h]
  char v41; // [rsp+15Ch] [rbp-134h]
  char v42; // [rsp+160h] [rbp-130h] BYREF

  if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
  {
    sub_B2C6D0(a2, a2, a3, a4);
    v4 = *(_QWORD *)(a2 + 96);
    result = v4 + 40LL * *(_QWORD *)(a2 + 104);
    v29 = result;
    if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
    {
      result = sub_B2C6D0(a2, a2, v26, v27);
      v4 = *(_QWORD *)(a2 + 96);
    }
  }
  else
  {
    v4 = *(_QWORD *)(a2 + 96);
    result = v4 + 40LL * *(_QWORD *)(a2 + 104);
    v29 = result;
  }
  if ( v4 != v29 )
  {
LABEL_6:
    result = sub_B2D680(v4);
    if ( !(_BYTE)result )
      goto LABEL_5;
    v37 = 0;
    v34 = (__int64 *)v36;
    v35 = 0x1000000000LL;
    v38 = (__int64 *)&v42;
    v32 = 0x1000000001LL;
    v8 = 1;
    v31 = v33;
    v39 = 32;
    v40 = 0;
    v41 = 1;
    v33[0] = v4;
    while ( 1 )
    {
      v9 = v31;
      v10 = v8;
      v11 = v31[v8 - 1];
      LODWORD(v32) = v8 - 1;
      if ( !v41 )
        goto LABEL_29;
      v12 = v38;
      v13 = HIDWORD(v39);
      v9 = &v38[HIDWORD(v39)];
      if ( v38 == v9 )
        break;
      while ( v11 != *v12 )
      {
        if ( v9 == ++v12 )
          goto LABEL_52;
      }
LABEL_13:
      v8 = v32;
      if ( !(_DWORD)v32 )
      {
        v14 = v34;
        result = (unsigned int)v35;
        v15 = (__int64)&v34[(unsigned int)v35];
        if ( v34 != (__int64 *)v15 )
        {
          v16 = *(unsigned __int8 *)(a1 + 28);
          while ( 1 )
          {
            while ( 1 )
            {
              v13 = *v14;
              if ( (_BYTE)v16 )
                break;
LABEL_54:
              v30 = (__int64 *)v15;
              ++v14;
              result = (unsigned __int64)sub_C8CC70(a1, v13, (__int64)v9, v15, v16, v7);
              v15 = (__int64)v30;
              v16 = *(unsigned __int8 *)(a1 + 28);
              if ( v30 == v14 )
                goto LABEL_22;
            }
            result = *(_QWORD *)(a1 + 8);
            v17 = *(unsigned int *)(a1 + 20);
            v9 = (__int64 *)(result + 8 * v17);
            if ( (__int64 *)result == v9 )
            {
LABEL_56:
              result = a1;
              if ( (unsigned int)v17 >= *(_DWORD *)(a1 + 16) )
                goto LABEL_54;
              ++v14;
              *(_DWORD *)(a1 + 20) = v17 + 1;
              *v9 = v13;
              v16 = *(unsigned __int8 *)(a1 + 28);
              ++*(_QWORD *)a1;
              if ( (__int64 *)v15 == v14 )
                break;
            }
            else
            {
              while ( v13 != *(_QWORD *)result )
              {
                result += 8LL;
                if ( v9 == (__int64 *)result )
                  goto LABEL_56;
              }
              if ( (__int64 *)v15 == ++v14 )
                break;
            }
          }
        }
LABEL_22:
        if ( !v41 )
          result = _libc_free(v38, v13);
        if ( v34 != (__int64 *)v36 )
          result = _libc_free(v34, v13);
        if ( v31 == v33 )
        {
LABEL_5:
          v4 += 40;
          if ( v29 == v4 )
            return result;
        }
        else
        {
          result = _libc_free(v31, v13);
          v4 += 40;
          if ( v29 == v4 )
            return result;
        }
        goto LABEL_6;
      }
    }
LABEL_52:
    if ( HIDWORD(v39) >= (unsigned int)v39 )
    {
LABEL_29:
      v13 = v11;
      sub_C8CC70((__int64)&v37, v11, (__int64)v9, v10, v6, v7);
      if ( !(_BYTE)v9 )
        goto LABEL_13;
    }
    else
    {
      v13 = (unsigned int)++HIDWORD(v39);
      *v9 = v11;
      ++v37;
    }
    v18 = *(_QWORD *)(v11 + 16);
    if ( !v18 )
      goto LABEL_13;
    while ( 1 )
    {
      v19 = *(_QWORD *)(v18 + 24);
      v20 = *(_BYTE *)v19;
      if ( *(_BYTE *)v19 <= 0x1Cu )
      {
LABEL_32:
        result = sub_CE8660(v4);
        if ( !(_BYTE)result )
          goto LABEL_22;
        goto LABEL_33;
      }
      if ( (unsigned __int8)(v20 - 78) <= 1u || v20 == 63 )
        goto LABEL_37;
      if ( v20 == 85 )
        break;
      if ( v20 != 61 )
        goto LABEL_32;
      v23 = (unsigned int)v35;
      v24 = (unsigned int)v35 + 1LL;
      if ( v24 > HIDWORD(v35) )
      {
        v13 = (__int64)v36;
        sub_C8D5F0((__int64)&v34, v36, v24, 8u, v6, v7);
        v23 = (unsigned int)v35;
      }
      v9 = v34;
      v34[v23] = v19;
      LODWORD(v35) = v35 + 1;
LABEL_33:
      v18 = *(_QWORD *)(v18 + 8);
      if ( !v18 )
        goto LABEL_13;
    }
    v25 = *(_QWORD *)(v19 - 32);
    if ( !v25 || *(_BYTE *)v25 || *(_QWORD *)(v25 + 24) != *(_QWORD *)(v19 + 80) || (*(_BYTE *)(v25 + 33) & 0x20) == 0 )
      goto LABEL_32;
    result = *(unsigned int *)(v25 + 36);
    if ( (_DWORD)result != 9250 && (_DWORD)result != 8923 )
      goto LABEL_22;
LABEL_37:
    v21 = (unsigned int)v32;
    v22 = (unsigned int)v32 + 1LL;
    if ( v22 > HIDWORD(v32) )
    {
      v13 = (__int64)v33;
      sub_C8D5F0((__int64)&v31, v33, v22, 8u, v6, v7);
      v21 = (unsigned int)v32;
    }
    v9 = v31;
    v31[v21] = v19;
    LODWORD(v32) = v32 + 1;
    goto LABEL_33;
  }
  return result;
}
