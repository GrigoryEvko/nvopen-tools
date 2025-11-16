// Function: sub_214CDF0
// Address: 0x214cdf0
//
__int64 __fastcall sub_214CDF0(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v4; // r13
  _QWORD *v5; // rax
  __int64 v6; // r14
  const char *v7; // r13
  _QWORD *v8; // rax
  int v9; // r9d
  __int64 *v10; // rbx
  _BYTE *v11; // r15
  size_t v12; // rax
  size_t v13; // r10
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 result; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  unsigned __int64 v21; // r15
  __int64 v22; // rax
  int v23; // r8d
  int v24; // r9d
  __int64 v25; // rdx
  unsigned __int64 v26; // r9
  size_t v27; // [rsp+8h] [rbp-78h]
  unsigned int v28; // [rsp+10h] [rbp-70h]
  __int64 v29; // [rsp+10h] [rbp-70h]
  _QWORD v31[2]; // [rsp+20h] [rbp-60h] BYREF
  _QWORD v32[2]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v33; // [rsp+40h] [rbp-40h]

  v4 = *(_QWORD *)(a1 + 264);
  v5 = *(_QWORD **)(v4 + 48);
  v6 = *(_QWORD *)(v4 + 8);
  if ( !v5 )
  {
    v19 = *(_QWORD *)(v4 + 120);
    v20 = *(_QWORD *)(v4 + 128);
    *(_QWORD *)(v4 + 200) += 280LL;
    if ( ((v19 + 7) & 0xFFFFFFFFFFFFFFF8LL) - v19 + 280 <= v20 - v19 )
    {
      v5 = (_QWORD *)((v19 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      *(_QWORD *)(v4 + 120) = v5 + 35;
    }
    else
    {
      v21 = 0x40000000000LL;
      v28 = *(_DWORD *)(v4 + 144);
      if ( v28 >> 7 < 0x1E )
        v21 = 4096LL << (v28 >> 7);
      v22 = malloc(v21);
      v25 = v28;
      if ( !v22 )
      {
        sub_16BD1C0("Allocation failed", 1u);
        v25 = *(unsigned int *)(v4 + 144);
        v22 = 0;
      }
      if ( (unsigned int)v25 >= *(_DWORD *)(v4 + 148) )
      {
        v29 = v22;
        sub_16CD150(v4 + 136, (const void *)(v4 + 152), 0, 8, v23, v24);
        v25 = *(unsigned int *)(v4 + 144);
        v22 = v29;
      }
      v26 = v22 + v21;
      *(_QWORD *)(*(_QWORD *)(v4 + 136) + 8 * v25) = v22;
      v5 = (_QWORD *)((v22 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      ++*(_DWORD *)(v4 + 144);
      *(_QWORD *)(v4 + 128) = v26;
      *(_QWORD *)(v4 + 120) = v5 + 35;
    }
    v5[2] = 0x800000000LL;
    *v5 = &unk_4A016E8;
    v5[1] = v5 + 3;
    *(_QWORD *)(v4 + 48) = v5;
  }
  v7 = *(const char **)(v5[1] + 32LL * a2);
  v8 = (_QWORD *)sub_22077B0(32);
  v10 = v8;
  if ( v8 )
  {
    v11 = v8 + 2;
    *v8 = v8 + 2;
    if ( !v7 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v12 = strlen(v7);
    v32[0] = v12;
    v13 = v12;
    if ( v12 > 0xF )
    {
      v27 = v12;
      v18 = sub_22409D0(v10, v32, 0);
      v13 = v27;
      *v10 = v18;
      v11 = (_BYTE *)v18;
      v10[2] = v32[0];
    }
    else
    {
      if ( v12 == 1 )
      {
        *((_BYTE *)v10 + 16) = *v7;
LABEL_7:
        v10[1] = v12;
        v11[v12] = 0;
        goto LABEL_8;
      }
      if ( !v12 )
        goto LABEL_7;
    }
    memcpy(v11, v7, v13);
    v12 = v32[0];
    v11 = (_BYTE *)*v10;
    goto LABEL_7;
  }
LABEL_8:
  v14 = *(unsigned int *)(v6 + 83296);
  if ( (unsigned int)v14 >= *(_DWORD *)(v6 + 83300) )
  {
    sub_16CD150(v6 + 83288, (const void *)(v6 + 83304), 0, 8, (int)v32, v9);
    v14 = *(unsigned int *)(v6 + 83296);
  }
  *(_QWORD *)(*(_QWORD *)(v6 + 83288) + 8 * v14) = v10;
  ++*(_DWORD *)(v6 + 83296);
  v15 = *(_QWORD *)(a1 + 248);
  v31[0] = *v10;
  v31[1] = v10[1];
  v33 = 261;
  v32[0] = v31;
  v16 = sub_38BF510(v15, v32);
  result = sub_38CF310(v16, 0, *(_QWORD *)(a1 + 248), 0);
  *(_BYTE *)a3 = 4;
  *(_QWORD *)(a3 + 8) = result;
  return result;
}
