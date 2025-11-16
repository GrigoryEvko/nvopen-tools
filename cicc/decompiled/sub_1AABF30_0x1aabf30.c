// Function: sub_1AABF30
// Address: 0x1aabf30
//
__int64 __fastcall sub_1AABF30(__int64 *a1, int a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // r15
  __int64 v8; // r14
  __int64 v9; // r13
  char v10; // di
  unsigned int v11; // esi
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // r12
  unsigned __int64 v16; // r9
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // r9
  __int64 v20; // rbx
  __int64 v21; // rdx
  __int64 v22; // r8
  __int64 v23; // r13
  __int64 *v24; // r12
  __int64 v25; // r14
  __int64 v26; // rcx
  int v27; // eax
  __int64 v28; // rax
  int v29; // edi
  __int64 v30; // rdi
  _QWORD *v31; // rax
  __int64 v32; // r9
  unsigned __int64 v33; // rdi
  __int64 v34; // rdi
  __int64 v35; // rdi
  __int64 v36; // rax
  __int64 *v37; // r13
  __int64 v38; // rdx
  unsigned __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // [rsp+0h] [rbp-90h]
  __int64 v43; // [rsp+10h] [rbp-80h]
  __int64 v44; // [rsp+18h] [rbp-78h]
  __int64 v45; // [rsp+20h] [rbp-70h]
  __int64 v46; // [rsp+28h] [rbp-68h]
  __int64 v48; // [rsp+38h] [rbp-58h]
  __int64 v49; // [rsp+38h] [rbp-58h]
  char *v50; // [rsp+40h] [rbp-50h] BYREF
  char v51; // [rsp+50h] [rbp-40h]
  char v52; // [rsp+51h] [rbp-3Fh]

  result = sub_157F280(a4);
  v43 = v6;
  if ( result != v6 )
  {
    v7 = result;
    v8 = a3;
    v41 = (__int64)&a1[(unsigned int)(a2 - 1) + 1];
    do
    {
      v9 = 0x17FFFFFFE8LL;
      v10 = *(_BYTE *)(v7 + 23) & 0x40;
      v11 = *(_DWORD *)(v7 + 20) & 0xFFFFFFF;
      if ( v11 )
      {
        v12 = 24LL * *(unsigned int *)(v7 + 56) + 8;
        v13 = 0;
        do
        {
          v14 = v7 - 24LL * v11;
          if ( v10 )
            v14 = *(_QWORD *)(v7 - 8);
          if ( v8 == *(_QWORD *)(v14 + v12) )
          {
            v9 = 24 * v13;
            goto LABEL_10;
          }
          ++v13;
          v12 += 8;
        }
        while ( v11 != (_DWORD)v13 );
        v9 = 0x17FFFFFFE8LL;
      }
LABEL_10:
      if ( v10 )
      {
        v15 = *(_QWORD *)(*(_QWORD *)(v7 - 8) + v9);
        if ( *(_BYTE *)(v15 + 16) != 77 )
          goto LABEL_12;
      }
      else
      {
        v15 = *(_QWORD *)(v7 - 24LL * v11 + v9);
        if ( *(_BYTE *)(v15 + 16) != 77 )
        {
LABEL_12:
          if ( !sub_157F790(v8) )
            goto LABEL_51;
          goto LABEL_13;
        }
      }
      if ( v8 != *(_QWORD *)(v15 + 40) )
      {
        if ( !sub_157F790(v8) )
        {
LABEL_51:
          v16 = sub_157EBA0(v8);
LABEL_15:
          v52 = 1;
          v50 = "split";
          v51 = 3;
          v17 = *(_QWORD *)v7;
          v46 = v16;
          v48 = *(_QWORD *)v7;
          v18 = sub_1648B60(64);
          v20 = v18;
          if ( v18 )
          {
            sub_15F1EA0(v18, v48, 53, 0, 0, v46);
            *(_DWORD *)(v20 + 56) = a2;
            sub_164B780(v20, (__int64 *)&v50);
            v17 = *(unsigned int *)(v20 + 56);
            sub_1648880(v20, v17, 1);
          }
          v21 = (__int64)a1;
          v22 = v15 + 8;
          if ( a2 )
          {
            v45 = v9;
            v23 = v15;
            v24 = a1;
            v44 = v8;
            v25 = v22;
            do
            {
              v26 = *v24;
              v27 = *(_DWORD *)(v20 + 20) & 0xFFFFFFF;
              if ( v27 == *(_DWORD *)(v20 + 56) )
              {
                v49 = *v24;
                sub_15F55D0(v20, v17, v21, v26, v22, v19);
                v26 = v49;
                v27 = *(_DWORD *)(v20 + 20) & 0xFFFFFFF;
              }
              v28 = (v27 + 1) & 0xFFFFFFF;
              v29 = v28 | *(_DWORD *)(v20 + 20) & 0xF0000000;
              *(_DWORD *)(v20 + 20) = v29;
              if ( (v29 & 0x40000000) != 0 )
                v30 = *(_QWORD *)(v20 - 8);
              else
                v30 = v20 - 24 * v28;
              v31 = (_QWORD *)(v30 + 24LL * (unsigned int)(v28 - 1));
              if ( *v31 )
              {
                v32 = v31[1];
                v33 = v31[2] & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v33 = v32;
                if ( v32 )
                  *(_QWORD *)(v32 + 16) = *(_QWORD *)(v32 + 16) & 3LL | v33;
              }
              *v31 = v23;
              v34 = *(_QWORD *)(v23 + 8);
              v31[1] = v34;
              if ( v34 )
                *(_QWORD *)(v34 + 16) = (unsigned __int64)(v31 + 1) | *(_QWORD *)(v34 + 16) & 3LL;
              v31[2] = v25 | v31[2] & 3LL;
              *(_QWORD *)(v23 + 8) = v31;
              v35 = *(_DWORD *)(v20 + 20) & 0xFFFFFFF;
              if ( (*(_BYTE *)(v20 + 23) & 0x40) != 0 )
                v19 = *(_QWORD *)(v20 - 8);
              else
                v19 = v20 - 24 * v35;
              ++v24;
              *(_QWORD *)(v19 + 8LL * (unsigned int)(v35 - 1) + 24LL * *(unsigned int *)(v20 + 56) + 8) = v26;
            }
            while ( v24 != (__int64 *)v41 );
            v9 = v45;
            v8 = v44;
          }
          if ( (*(_BYTE *)(v7 + 23) & 0x40) != 0 )
            v36 = *(_QWORD *)(v7 - 8);
          else
            v36 = v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF);
          v37 = (__int64 *)(v36 + v9);
          if ( *v37 )
          {
            v38 = v37[1];
            v39 = v37[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v39 = v38;
            if ( v38 )
              *(_QWORD *)(v38 + 16) = *(_QWORD *)(v38 + 16) & 3LL | v39;
          }
          *v37 = v20;
          if ( v20 )
          {
            v40 = *(_QWORD *)(v20 + 8);
            v37[1] = v40;
            if ( v40 )
              *(_QWORD *)(v40 + 16) = (unsigned __int64)(v37 + 1) | *(_QWORD *)(v40 + 16) & 3LL;
            v37[2] = (v20 + 8) | v37[2] & 3;
            *(_QWORD *)(v20 + 8) = v37;
          }
          goto LABEL_43;
        }
LABEL_13:
        v16 = *(_QWORD *)(v8 + 48);
        if ( v16 )
          v16 -= 24LL;
        goto LABEL_15;
      }
LABEL_43:
      result = *(_QWORD *)(v7 + 32);
      if ( !result )
        BUG();
      v7 = 0;
      if ( *(_BYTE *)(result - 8) == 77 )
        v7 = result - 24;
    }
    while ( v43 != v7 );
  }
  return result;
}
