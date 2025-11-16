// Function: sub_2576560
// Address: 0x2576560
//
__int64 __fastcall sub_2576560(__int64 a1, __int64 a2)
{
  __int64 (*v4)(void); // rax
  __int64 result; // rax
  __int64 (__fastcall *v6)(__int64); // rax
  char v7; // al
  __int64 v8; // rbx
  const void **v9; // r14
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // r9
  const void **v13; // r13
  __int64 v14; // rdi
  unsigned __int64 v15; // rcx
  __int64 v16; // rsi
  int v17; // eax
  __int64 v18; // rdi
  unsigned int v19; // eax
  unsigned int v20; // eax
  unsigned int v21; // edx
  __int64 (__fastcall *v22)(__int64); // rax
  __int64 v23; // rsi
  unsigned int v24; // eax
  char *v25; // r13
  __int64 v26; // [rsp+8h] [rbp-88h]
  const void **v27; // [rsp+10h] [rbp-80h]
  __int64 v28; // [rsp+18h] [rbp-78h]
  unsigned __int64 v29; // [rsp+20h] [rbp-70h]
  unsigned __int64 v30; // [rsp+28h] [rbp-68h]
  _BYTE v31[96]; // [rsp+30h] [rbp-60h] BYREF

  v4 = *(__int64 (**)(void))(*(_QWORD *)a1 + 16LL);
  if ( (char *)v4 == (char *)sub_2505E40 )
    result = *(unsigned __int8 *)(a1 + 17);
  else
    result = v4();
  if ( (_BYTE)result )
  {
    v6 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 16LL);
    if ( v6 == sub_2505E40 )
      v7 = *(_BYTE *)(a2 + 17);
    else
      v7 = v6(a2);
    if ( !v7 )
    {
LABEL_22:
      v22 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 40LL);
      if ( v22 != sub_2534FB0 )
        return v22(a1);
      result = *(unsigned __int8 *)(a1 + 16);
      *(_BYTE *)(a1 + 17) = result;
      return result;
    }
    v26 = a1 + 24;
    v8 = 2LL * *(unsigned int *)(a2 + 64);
    v9 = *(const void ***)(a2 + 56);
    v27 = &v9[v8];
    if ( &v9[v8] == v9 )
    {
LABEL_19:
      v20 = *(unsigned __int8 *)(a1 + 200);
      LOBYTE(v20) = *(_BYTE *)(a2 + 200) | v20;
      v21 = *(_DWORD *)(a1 + 64);
      *(_BYTE *)(a1 + 200) = v20;
      if ( v21 < unk_4FEF868 )
      {
        LOBYTE(v21) = v21 == 0;
        result = v21 & v20;
        *(_BYTE *)(a1 + 200) = result;
        return result;
      }
      goto LABEL_22;
    }
    while ( 1 )
    {
      v13 = v9;
      if ( !*(_DWORD *)(a1 + 40) )
        break;
      sub_2575C40((__int64)v31, v26, (__int64)v9);
      if ( v31[32] )
      {
        v14 = *(unsigned int *)(a1 + 64);
        v15 = *(_QWORD *)(a1 + 56);
        v16 = v14 + 1;
        v17 = *(_DWORD *)(a1 + 64);
        if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 68) )
        {
          if ( v15 > (unsigned __int64)v9 || (unsigned __int64)v9 >= v15 + 16 * v14 )
          {
            sub_AE4800((unsigned int *)(a1 + 56), v16);
            v14 = *(unsigned int *)(a1 + 64);
            v15 = *(_QWORD *)(a1 + 56);
            v17 = *(_DWORD *)(a1 + 64);
          }
          else
          {
            v25 = (char *)v9 - v15;
            sub_AE4800((unsigned int *)(a1 + 56), v16);
            v15 = *(_QWORD *)(a1 + 56);
            v14 = *(unsigned int *)(a1 + 64);
            v13 = (const void **)&v25[v15];
            v17 = *(_DWORD *)(a1 + 64);
          }
        }
        v18 = v15 + 16 * v14;
        if ( v18 )
        {
          v19 = *((_DWORD *)v13 + 2);
          *(_DWORD *)(v18 + 8) = v19;
          if ( v19 > 0x40 )
            sub_C43780(v18, v13);
          else
            *(_QWORD *)v18 = *v13;
          v17 = *(_DWORD *)(a1 + 64);
        }
        v9 += 2;
        *(_DWORD *)(a1 + 64) = v17 + 1;
        if ( v27 == v9 )
          goto LABEL_19;
      }
      else
      {
LABEL_10:
        v9 += 2;
        if ( v27 == v9 )
          goto LABEL_19;
      }
    }
    v28 = *(unsigned int *)(a1 + 64);
    LODWORD(v10) = v28;
    v30 = *(_QWORD *)(a1 + 56);
    v29 = v30 + 16 * v28;
    v11 = sub_2546E70(v30, v29, v9);
    v12 = v29;
    if ( v29 == v11 )
    {
      v23 = v28 + 1;
      if ( v28 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 68) )
      {
        if ( v30 > (unsigned __int64)v9 || v29 <= (unsigned __int64)v9 )
        {
          sub_AE4800((unsigned int *)(a1 + 56), v23);
          v10 = *(unsigned int *)(a1 + 64);
          v12 = *(_QWORD *)(a1 + 56) + 16 * v10;
        }
        else
        {
          sub_AE4800((unsigned int *)(a1 + 56), v23);
          v13 = (const void **)((char *)v9 + *(_QWORD *)(a1 + 56) - v30);
          v10 = *(unsigned int *)(a1 + 64);
          v12 = *(_QWORD *)(a1 + 56) + 16 * v10;
        }
      }
      if ( v12 )
      {
        v24 = *((_DWORD *)v13 + 2);
        *(_DWORD *)(v12 + 8) = v24;
        if ( v24 > 0x40 )
          sub_C43780(v12, v13);
        else
          *(_QWORD *)v12 = *v13;
        LODWORD(v10) = *(_DWORD *)(a1 + 64);
      }
      *(_DWORD *)(a1 + 64) = v10 + 1;
      if ( (unsigned int)(v10 + 1) > 8 )
        sub_2575D90(v26);
    }
    goto LABEL_10;
  }
  return result;
}
