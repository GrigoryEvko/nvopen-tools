// Function: sub_F34B50
// Address: 0xf34b50
//
__int64 __fastcall sub_F34B50(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v5; // r13
  __int64 v6; // r14
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r12
  __int64 v11; // r12
  __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // rsi
  __int64 *v18; // r14
  __int64 v19; // r15
  __int64 v20; // rbx
  __int64 *v21; // r12
  int v22; // eax
  unsigned int v23; // edi
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rdi
  __int64 v27; // r13
  int v28; // eax
  __int64 *v29; // r12
  __int64 v30; // rax
  __int64 v31; // rax
  unsigned __int64 v32; // rax
  unsigned __int64 v33; // rsi
  __int64 v34; // [rsp+8h] [rbp-A8h]
  int v35; // [rsp+10h] [rbp-A0h]
  __int64 v37; // [rsp+20h] [rbp-90h]
  __int64 v38; // [rsp+28h] [rbp-88h]
  __int64 v39; // [rsp+30h] [rbp-80h]
  __int64 v40; // [rsp+48h] [rbp-68h]
  char *v41; // [rsp+50h] [rbp-60h] BYREF
  char v42; // [rsp+70h] [rbp-40h]
  char v43; // [rsp+71h] [rbp-3Fh]

  v5 = &a1[a2];
  v35 = a2;
  v6 = sub_AA5930(a4);
  result = a3 + 48;
  v37 = v8;
  v34 = a3 + 48;
  while ( v37 != v6 )
  {
    v9 = *(_QWORD *)(v6 - 8);
    v40 = 0x1FFFFFFFE0LL;
    if ( (*(_DWORD *)(v6 + 4) & 0x7FFFFFF) != 0 )
    {
      v10 = 0;
      do
      {
        if ( a3 == *(_QWORD *)(v9 + 32LL * *(unsigned int *)(v6 + 72) + 8 * v10) )
        {
          v40 = 32 * v10;
          goto LABEL_7;
        }
        ++v10;
      }
      while ( (*(_DWORD *)(v6 + 4) & 0x7FFFFFF) != (_DWORD)v10 );
      v40 = 0x1FFFFFFFE0LL;
      v11 = *(_QWORD *)(v9 + 0x1FFFFFFFE0LL);
      if ( *(_BYTE *)v11 != 84 )
        goto LABEL_8;
    }
    else
    {
LABEL_7:
      v11 = *(_QWORD *)(v9 + v40);
      if ( *(_BYTE *)v11 != 84 )
        goto LABEL_8;
    }
    if ( a3 == *(_QWORD *)(v11 + 40) )
      goto LABEL_26;
LABEL_8:
    v43 = 1;
    v42 = 3;
    v12 = *(_QWORD *)(v6 + 8);
    v41 = "split";
    v13 = sub_BD2DA0(80);
    v14 = v13;
    if ( v13 )
    {
      sub_B44260(v13, v12, 55, 0x8000000u, 0, 0);
      *(_DWORD *)(v14 + 72) = v35;
      sub_BD6B50((unsigned __int8 *)v14, (const char **)&v41);
      sub_BD2A10(v14, *(_DWORD *)(v14 + 72), 1);
    }
    if ( sub_AA5E90(a3) )
    {
      v15 = *(_QWORD *)(a3 + 56);
      v16 = 1;
    }
    else
    {
      v32 = *(_QWORD *)(a3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v32 == v34 )
      {
        v33 = 0;
      }
      else
      {
        if ( !v32 )
          BUG();
        v33 = v32 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v32 - 24) - 30 >= 0xB )
          v33 = 0;
      }
      v15 = v33 + 24;
      v16 = 0;
    }
    sub_B44220((_QWORD *)v14, v15, v16);
    if ( a1 != v5 )
    {
      v39 = v6;
      v17 = v11 + 16;
      v18 = v5;
      v38 = a3;
      v19 = v14;
      v20 = v11;
      v21 = a1;
      do
      {
        v27 = *v21;
        v28 = *(_DWORD *)(v19 + 4) & 0x7FFFFFF;
        if ( v28 == *(_DWORD *)(v19 + 72) )
        {
          sub_B48D90(v19);
          v28 = *(_DWORD *)(v19 + 4) & 0x7FFFFFF;
        }
        v22 = (v28 + 1) & 0x7FFFFFF;
        v23 = v22 | *(_DWORD *)(v19 + 4) & 0xF8000000;
        v24 = *(_QWORD *)(v19 - 8) + 32LL * (unsigned int)(v22 - 1);
        *(_DWORD *)(v19 + 4) = v23;
        if ( *(_QWORD *)v24 )
        {
          v25 = *(_QWORD *)(v24 + 8);
          **(_QWORD **)(v24 + 16) = v25;
          if ( v25 )
            *(_QWORD *)(v25 + 16) = *(_QWORD *)(v24 + 16);
        }
        *(_QWORD *)v24 = v20;
        v26 = *(_QWORD *)(v20 + 16);
        *(_QWORD *)(v24 + 8) = v26;
        if ( v26 )
          *(_QWORD *)(v26 + 16) = v24 + 8;
        *(_QWORD *)(v24 + 16) = v17;
        ++v21;
        *(_QWORD *)(v20 + 16) = v24;
        *(_QWORD *)(*(_QWORD *)(v19 - 8)
                  + 32LL * *(unsigned int *)(v19 + 72)
                  + 8LL * ((*(_DWORD *)(v19 + 4) & 0x7FFFFFFu) - 1)) = v27;
      }
      while ( v18 != v21 );
      v5 = v18;
      v6 = v39;
      v14 = v19;
      a3 = v38;
      v29 = (__int64 *)(*(_QWORD *)(v39 - 8) + v40);
      if ( !*v29 || (v30 = v29[1], (*(_QWORD *)v29[2] = v30) == 0) )
      {
        *v29 = v14;
LABEL_32:
        v31 = *(_QWORD *)(v14 + 16);
        v29[1] = v31;
        if ( v31 )
          *(_QWORD *)(v31 + 16) = v29 + 1;
        v29[2] = v14 + 16;
        *(_QWORD *)(v14 + 16) = v29;
        goto LABEL_26;
      }
LABEL_24:
      *(_QWORD *)(v30 + 16) = v29[2];
      goto LABEL_25;
    }
    v29 = (__int64 *)(*(_QWORD *)(v6 - 8) + v40);
    if ( *v29 )
    {
      v30 = v29[1];
      *(_QWORD *)v29[2] = v30;
      if ( v30 )
        goto LABEL_24;
    }
LABEL_25:
    *v29 = v14;
    if ( v14 )
      goto LABEL_32;
LABEL_26:
    result = *(_QWORD *)(v6 + 32);
    if ( !result )
      BUG();
    v6 = 0;
    if ( *(_BYTE *)(result - 24) == 84 )
      v6 = result - 24;
  }
  return result;
}
