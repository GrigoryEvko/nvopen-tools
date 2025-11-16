// Function: sub_146A580
// Address: 0x146a580
//
__int64 __fastcall sub_146A580(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r12
  int v9; // eax
  __int64 v10; // rsi
  int v11; // ecx
  unsigned int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // rdi
  __int64 result; // rax
  int v16; // eax
  size_t v17; // r14
  int v18; // eax
  __int64 v19; // r15
  __int64 v20; // r13
  __int64 v21; // r12
  __int64 *v22; // r14
  __int64 *v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rax
  unsigned int v26; // esi
  __int64 v27; // rbx
  __int64 v28; // rcx
  __int64 v29; // r8
  unsigned int v30; // edx
  __int64 *v31; // rax
  __int64 v32; // rdi
  __int64 v33; // rax
  char v34; // al
  int v35; // r8d
  int v36; // r11d
  __int64 *v37; // r9
  int v38; // edi
  int v39; // edi
  _QWORD *v40; // [rsp+18h] [rbp-78h]
  size_t v41; // [rsp+20h] [rbp-70h]
  __int64 v42; // [rsp+28h] [rbp-68h]
  _QWORD *v43; // [rsp+30h] [rbp-60h]
  __int64 v44; // [rsp+38h] [rbp-58h]
  __int64 v45; // [rsp+40h] [rbp-50h]
  __int64 v47; // [rsp+48h] [rbp-48h]
  __int64 v48; // [rsp+50h] [rbp-40h] BYREF
  __int64 *v49; // [rsp+58h] [rbp-38h] BYREF

  v7 = a1;
  v9 = *(_DWORD *)(a3 + 24);
  if ( v9 )
  {
    v10 = *(_QWORD *)(a3 + 8);
    v11 = v9 - 1;
    v12 = (v9 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v13 = (__int64 *)(v10 + 16LL * v12);
    v14 = *v13;
    if ( v7 == *v13 )
    {
LABEL_3:
      result = v13[1];
      if ( result )
        return result;
    }
    else
    {
      v16 = 1;
      while ( v14 != -8 )
      {
        v35 = v16 + 1;
        v12 = v11 & (v16 + v12);
        v13 = (__int64 *)(v10 + 16LL * v12);
        v14 = *v13;
        if ( v7 == *v13 )
          goto LABEL_3;
        v16 = v35;
      }
    }
  }
  if ( !sub_1377F70(a2 + 56, *(_QWORD *)(v7 + 40)) )
    return 0;
  result = 0;
  if ( *(_BYTE *)(v7 + 16) == 77 )
    return result;
  if ( !(unsigned __int8)sub_1452C00(v7) || *(_BYTE *)(v7 + 16) == 77 )
    return 0;
  if ( (*(_DWORD *)(v7 + 20) & 0xFFFFFFF) != 0 )
  {
    v17 = 8LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF);
    v43 = (_QWORD *)sub_22077B0(v17);
    v40 = &v43[v17 / 8];
    memset(v43, 0, v17);
    v41 = v17;
    v18 = *(_DWORD *)(v7 + 20) & 0xFFFFFFF;
    if ( v18 )
    {
      v42 = a2;
      v44 = a4;
      v19 = 0;
      v20 = v7;
      v21 = a3;
      v22 = v43;
      v45 = (__int64)&v43[(unsigned int)(v18 - 1) + 1];
      while ( 1 )
      {
        if ( (*(_BYTE *)(v20 + 23) & 0x40) != 0 )
        {
          v23 = (__int64 *)(v19 + *(_QWORD *)(v20 - 8));
          v24 = *v23;
          if ( *(_BYTE *)(*v23 + 16) > 0x17u )
            goto LABEL_16;
        }
        else
        {
          v23 = (__int64 *)(v19 + v20 - 24LL * (*(_DWORD *)(v20 + 20) & 0xFFFFFFF));
          v24 = *v23;
          if ( *(_BYTE *)(*v23 + 16) > 0x17u )
          {
LABEL_16:
            v48 = v24;
            v25 = sub_146A580(v24, v42, v21, v44, a5);
            v26 = *(_DWORD *)(v21 + 24);
            v27 = v25;
            if ( v26 )
            {
              v28 = v48;
              v29 = *(_QWORD *)(v21 + 8);
              v30 = (v26 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
              v31 = (__int64 *)(v29 + 16LL * v30);
              v32 = *v31;
              if ( *v31 == v48 )
              {
LABEL_18:
                v31[1] = v27;
                if ( !v27 )
                  goto LABEL_27;
                *v22 = v27;
                goto LABEL_20;
              }
              v36 = 1;
              v37 = 0;
              while ( v32 != -8 )
              {
                if ( !v37 && v32 == -16 )
                  v37 = v31;
                v30 = (v26 - 1) & (v36 + v30);
                v31 = (__int64 *)(v29 + 16LL * v30);
                v32 = *v31;
                if ( v48 == *v31 )
                  goto LABEL_18;
                ++v36;
              }
              v38 = *(_DWORD *)(v21 + 16);
              if ( v37 )
                v31 = v37;
              ++*(_QWORD *)v21;
              v39 = v38 + 1;
              if ( 4 * v39 < 3 * v26 )
              {
                if ( v26 - *(_DWORD *)(v21 + 20) - v39 > v26 >> 3 )
                {
LABEL_44:
                  *(_DWORD *)(v21 + 16) = v39;
                  if ( *v31 != -8 )
                    --*(_DWORD *)(v21 + 20);
                  *v31 = v28;
                  v31[1] = 0;
                  goto LABEL_18;
                }
LABEL_49:
                sub_146A3C0(v21, v26);
                sub_1463C30(v21, &v48, &v49);
                v31 = v49;
                v28 = v48;
                v39 = *(_DWORD *)(v21 + 16) + 1;
                goto LABEL_44;
              }
            }
            else
            {
              ++*(_QWORD *)v21;
            }
            v26 *= 2;
            goto LABEL_49;
          }
        }
        v48 = 0;
        v33 = *v23;
        if ( *(_BYTE *)(v33 + 16) > 0x10u )
        {
          *v22 = 0;
LABEL_27:
          result = 0;
          goto LABEL_28;
        }
        *v22 = v33;
LABEL_20:
        ++v22;
        v19 += 24;
        if ( v22 == (__int64 *)v45 )
        {
          v7 = v20;
          a4 = v44;
          break;
        }
      }
    }
  }
  else
  {
    v43 = 0;
    v41 = 0;
    v40 = 0;
  }
  v34 = *(_BYTE *)(v7 + 16);
  if ( (unsigned __int8)(v34 - 75) <= 1u )
  {
    result = sub_14D7760(*(_WORD *)(v7 + 18) & 0x7FFF, *v43, v43[1], a4, a5);
    goto LABEL_28;
  }
  if ( v34 == 54 && (*(_BYTE *)(v7 + 18) & 1) == 0 )
  {
    result = sub_14D8290(*v43, *(_QWORD *)v7, a4);
    goto LABEL_28;
  }
  result = sub_14DD1F0(v7, v43, v40 - v43, a4, a5);
  if ( v43 )
  {
LABEL_28:
    v47 = result;
    j_j___libc_free_0(v43, v41);
    return v47;
  }
  return result;
}
