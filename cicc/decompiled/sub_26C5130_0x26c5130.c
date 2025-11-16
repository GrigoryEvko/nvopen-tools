// Function: sub_26C5130
// Address: 0x26c5130
//
__int64 *__fastcall sub_26C5130(__int64 a1, __int64 *a2)
{
  int v3; // edx
  __int64 *v4; // rcx
  __int64 v5; // r13
  unsigned int v6; // esi
  __int64 *v7; // rax
  __int64 v8; // r10
  int v9; // eax
  __int64 v10; // rax
  unsigned int v11; // r8d
  __int64 v12; // rdi
  __int64 *result; // rax
  __int64 *v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rsi
  __int64 v17; // rcx
  __int64 v18; // rbx
  __int64 v19; // r15
  __int64 v20; // r14
  _BYTE *v21; // rsi
  unsigned int v22; // edx
  __int64 v23; // rsi
  unsigned int v24; // ecx
  __int64 *v25; // rax
  __int64 v26; // r8
  int v27; // ecx
  int v28; // eax
  __int64 v29; // rdi
  unsigned int v30; // r8d
  unsigned int v31; // r10d
  __int64 *v32; // rax
  __int64 v33; // rbx
  int v34; // eax
  int v35; // eax
  int v36; // r11d
  int v37; // eax
  int v38; // eax
  int v39; // r11d
  int v40; // r9d
  __int64 v42; // [rsp+10h] [rbp-50h]
  __int64 v43; // [rsp+18h] [rbp-48h]
  __int64 v44[7]; // [rsp+28h] [rbp-38h] BYREF

  sub_D2AD40(a1, a2);
  v3 = *(_DWORD *)(a1 + 440);
  if ( v3 )
  {
    v4 = *(__int64 **)(a1 + 432);
    v5 = *v4;
    if ( *v4 )
    {
      while ( 1 )
      {
        v10 = *(unsigned int *)(v5 + 16);
        if ( (_DWORD)v10 )
          break;
        v11 = *(_DWORD *)(a1 + 600);
        v12 = *(_QWORD *)(a1 + 584);
        if ( v11 )
        {
          v6 = (v11 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
          v7 = (__int64 *)(v12 + 16LL * v6);
          v8 = *v7;
          if ( *v7 == v5 )
          {
LABEL_5:
            v9 = *((_DWORD *)v7 + 2) + 1;
            if ( v9 == v3 )
              goto LABEL_10;
            goto LABEL_6;
          }
          v38 = 1;
          while ( v8 != -4096 )
          {
            v39 = v38 + 1;
            v6 = (v11 - 1) & (v38 + v6);
            v7 = (__int64 *)(v12 + 16LL * v6);
            v8 = *v7;
            if ( *v7 == v5 )
              goto LABEL_5;
            v38 = v39;
          }
        }
        v9 = *(_DWORD *)(v12 + 16LL * v11 + 8) + 1;
        if ( v9 == v3 )
          goto LABEL_10;
LABEL_6:
        v5 = v4[v9];
        if ( !v5 )
          goto LABEL_10;
      }
LABEL_14:
      v43 = *(_QWORD *)(v5 + 8);
      v42 = v43 + 8 * v10;
      while ( 1 )
      {
        v17 = *(_QWORD *)(*(_QWORD *)v43 + 8LL);
        v18 = v17 + 8LL * *(unsigned int *)(*(_QWORD *)v43 + 16LL);
        v19 = v17;
        if ( v18 != v17 )
          break;
LABEL_24:
        v43 += 8;
        if ( v42 == v43 )
        {
          v22 = *(_DWORD *)(a1 + 600);
          v23 = *(_QWORD *)(a1 + 584);
          if ( v22 )
          {
            v24 = (v22 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
            v25 = (__int64 *)(v23 + 16LL * v24);
            v26 = *v25;
            if ( *v25 == v5 )
              goto LABEL_27;
            v37 = 1;
            while ( v26 != -4096 )
            {
              v40 = v37 + 1;
              v24 = (v22 - 1) & (v37 + v24);
              v25 = (__int64 *)(v23 + 16LL * v24);
              v26 = *v25;
              if ( *v25 == v5 )
                goto LABEL_27;
              v37 = v40;
            }
          }
          v25 = (__int64 *)(v23 + 16LL * v22);
LABEL_27:
          v27 = *(_DWORD *)(a1 + 440);
          v28 = *((_DWORD *)v25 + 2) + 1;
          if ( v28 == v27 )
            goto LABEL_10;
          v29 = *(_QWORD *)(a1 + 432);
          v5 = *(_QWORD *)(v29 + 8LL * v28);
          if ( !v5 )
            goto LABEL_10;
          v30 = v22 - 1;
          while ( 1 )
          {
            v10 = *(unsigned int *)(v5 + 16);
            if ( (_DWORD)v10 )
              goto LABEL_14;
            if ( v22 )
            {
              v31 = v30 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
              v32 = (__int64 *)(v23 + 16LL * v31);
              v33 = *v32;
              if ( v5 == *v32 )
                goto LABEL_31;
              v35 = 1;
              while ( v33 != -4096 )
              {
                v36 = v35 + 1;
                v31 = v30 & (v35 + v31);
                v32 = (__int64 *)(v23 + 16LL * v31);
                v33 = *v32;
                if ( v5 == *v32 )
                  goto LABEL_31;
                v35 = v36;
              }
            }
            v32 = (__int64 *)(v23 + 16LL * v22);
LABEL_31:
            v34 = *((_DWORD *)v32 + 2) + 1;
            if ( v27 != v34 )
            {
              v5 = *(_QWORD *)(v29 + 8LL * v34);
              if ( v5 )
                continue;
            }
            goto LABEL_10;
          }
        }
      }
      while ( 1 )
      {
        v20 = *(_QWORD *)(*(_QWORD *)v19 + 8LL);
        if ( sub_B2FC80(v20) || !(unsigned __int8)sub_B2D620(v20, "use-sample-profile", 0x12u) )
          goto LABEL_17;
        v44[0] = v20;
        v21 = (_BYTE *)a2[1];
        if ( v21 == (_BYTE *)a2[2] )
        {
          sub_24147A0((__int64)a2, v21, v44);
LABEL_17:
          v19 += 8;
          if ( v18 == v19 )
            goto LABEL_24;
        }
        else
        {
          if ( v21 )
          {
            *(_QWORD *)v21 = v20;
            v21 = (_BYTE *)a2[1];
          }
          v19 += 8;
          a2[1] = (__int64)(v21 + 8);
          if ( v18 == v19 )
            goto LABEL_24;
        }
      }
    }
  }
LABEL_10:
  result = (__int64 *)a2[1];
  v14 = (__int64 *)*a2;
  if ( (__int64 *)*a2 != result )
  {
    for ( --result; result > v14; result[1] = v15 )
    {
      v15 = *v14;
      v16 = *result;
      ++v14;
      --result;
      *(v14 - 1) = v16;
    }
  }
  return result;
}
