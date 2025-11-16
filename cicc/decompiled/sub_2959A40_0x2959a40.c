// Function: sub_2959A40
// Address: 0x2959a40
//
const char **__fastcall sub_2959A40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  __int64 v6; // r14
  const char **result; // rax
  __int64 v8; // rdx
  const char *v9; // rdx
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // r15
  _QWORD *v13; // r12
  int v14; // esi
  signed int v15; // r12d
  __int64 v16; // rax
  __int64 v17; // rbx
  __int64 v18; // r13
  __int64 v19; // rdx
  __int64 v20; // r8
  int v21; // eax
  int v22; // eax
  unsigned int v23; // edx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rdx
  int v27; // eax
  int v28; // eax
  unsigned int v29; // edx
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // [rsp+8h] [rbp-98h]
  __int64 v34; // [rsp+10h] [rbp-90h]
  __int64 v35; // [rsp+28h] [rbp-78h]
  __int64 v36; // [rsp+28h] [rbp-78h]
  const char *v39[4]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v40; // [rsp+60h] [rbp-40h]

  v33 = *(_QWORD *)(a2 + 56);
  v6 = sub_AA5930(a1);
  result = v39;
  v34 = v8;
  if ( v8 != v6 )
  {
    while ( 1 )
    {
      v39[0] = sub_BD5D20(v6);
      v40 = 773;
      v39[1] = v9;
      v39[2] = ".split";
      v10 = *(_QWORD *)(v6 + 8);
      v11 = sub_BD2DA0(80);
      v12 = v11;
      if ( v11 )
      {
        v13 = (_QWORD *)v11;
        sub_B44260(v11, v10, 55, 0x8000000u, 0, 0);
        *(_DWORD *)(v12 + 72) = 2;
        sub_BD6B50((unsigned __int8 *)v12, v39);
        sub_BD2A10(v12, *(_DWORD *)(v12 + 72), 1);
      }
      else
      {
        v13 = 0;
      }
      sub_B44220(v13, v33, 1);
      v14 = *(_DWORD *)(v6 + 4) & 0x7FFFFFF;
      v15 = v14 - 1;
      if ( v14 )
        break;
LABEL_19:
      sub_BD84D0(v6, v12);
      v27 = *(_DWORD *)(v12 + 4) & 0x7FFFFFF;
      if ( v27 == *(_DWORD *)(v12 + 72) )
      {
        sub_B48D90(v12);
        v27 = *(_DWORD *)(v12 + 4) & 0x7FFFFFF;
      }
      v28 = (v27 + 1) & 0x7FFFFFF;
      v29 = v28 | *(_DWORD *)(v12 + 4) & 0xF8000000;
      v30 = *(_QWORD *)(v12 - 8) + 32LL * (unsigned int)(v28 - 1);
      *(_DWORD *)(v12 + 4) = v29;
      if ( *(_QWORD *)v30 )
      {
        v31 = *(_QWORD *)(v30 + 8);
        **(_QWORD **)(v30 + 16) = v31;
        if ( v31 )
          *(_QWORD *)(v31 + 16) = *(_QWORD *)(v30 + 16);
      }
      *(_QWORD *)v30 = v6;
      if ( v6 )
      {
        v32 = *(_QWORD *)(v6 + 16);
        *(_QWORD *)(v30 + 8) = v32;
        if ( v32 )
          *(_QWORD *)(v32 + 16) = v30 + 8;
        *(_QWORD *)(v30 + 16) = v6 + 16;
        *(_QWORD *)(v6 + 16) = v30;
      }
      *(_QWORD *)(*(_QWORD *)(v12 - 8)
                + 32LL * *(unsigned int *)(v12 + 72)
                + 8LL * ((*(_DWORD *)(v12 + 4) & 0x7FFFFFFu) - 1)) = a1;
      result = *(const char ***)(v6 + 32);
      if ( !result )
        BUG();
      v6 = 0;
      if ( *((_BYTE *)result - 24) == 84 )
        v6 = (__int64)(result - 3);
      if ( v34 == v6 )
        return result;
    }
    v16 = a3;
    v17 = 8LL * v15;
    v18 = v16;
    while ( 1 )
    {
      while ( 1 )
      {
        v19 = *(_QWORD *)(v6 - 8);
        if ( v18 == *(_QWORD *)(v19 + 32LL * *(unsigned int *)(v6 + 72) + v17) )
          break;
        --v15;
        v17 -= 8;
        if ( v15 == -1 )
          goto LABEL_18;
      }
      v20 = *(_QWORD *)(v19 + 4 * v17);
      if ( a5 )
      {
        v35 = *(_QWORD *)(v19 + 4 * v17);
        sub_B48BF0(v6, v15, 1);
        v20 = v35;
        v21 = *(_DWORD *)(v12 + 4) & 0x7FFFFFF;
        if ( v21 != *(_DWORD *)(v12 + 72) )
          goto LABEL_10;
      }
      else
      {
        v21 = *(_DWORD *)(v12 + 4) & 0x7FFFFFF;
        if ( v21 != *(_DWORD *)(v12 + 72) )
          goto LABEL_10;
      }
      v36 = v20;
      sub_B48D90(v12);
      v20 = v36;
      v21 = *(_DWORD *)(v12 + 4) & 0x7FFFFFF;
LABEL_10:
      v22 = (v21 + 1) & 0x7FFFFFF;
      v23 = v22 | *(_DWORD *)(v12 + 4) & 0xF8000000;
      v24 = *(_QWORD *)(v12 - 8) + 32LL * (unsigned int)(v22 - 1);
      *(_DWORD *)(v12 + 4) = v23;
      if ( *(_QWORD *)v24 )
      {
        v25 = *(_QWORD *)(v24 + 8);
        **(_QWORD **)(v24 + 16) = v25;
        if ( v25 )
          *(_QWORD *)(v25 + 16) = *(_QWORD *)(v24 + 16);
      }
      *(_QWORD *)v24 = v20;
      if ( v20 )
      {
        v26 = *(_QWORD *)(v20 + 16);
        *(_QWORD *)(v24 + 8) = v26;
        if ( v26 )
          *(_QWORD *)(v26 + 16) = v24 + 8;
        *(_QWORD *)(v24 + 16) = v20 + 16;
        *(_QWORD *)(v20 + 16) = v24;
      }
      --v15;
      v17 -= 8;
      *(_QWORD *)(*(_QWORD *)(v12 - 8)
                + 32LL * *(unsigned int *)(v12 + 72)
                + 8LL * ((*(_DWORD *)(v12 + 4) & 0x7FFFFFFu) - 1)) = a4;
      if ( v15 == -1 )
      {
LABEL_18:
        a3 = v18;
        goto LABEL_19;
      }
    }
  }
  return result;
}
