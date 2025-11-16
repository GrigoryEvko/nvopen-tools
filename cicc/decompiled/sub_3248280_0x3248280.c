// Function: sub_3248280
// Address: 0x3248280
//
__int64 __fastcall sub_3248280(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rbx
  __int64 result; // rax
  __int64 v9; // rdx
  __int64 v10; // rdi
  __int64 v11; // rdx
  unsigned __int8 v12; // al
  __int64 v13; // r11
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r8
  unsigned __int8 v19; // al
  __int64 v20; // r8
  size_t v21; // rdx
  size_t v22; // r9
  __int64 v23; // rdx
  __int64 v24; // rdi
  const void *v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  unsigned __int8 v28; // al
  __int64 v29; // r11
  __int64 v30; // rbx
  __int64 v31; // rcx
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // r8
  void *s2; // [rsp+8h] [rbp-48h]
  size_t na; // [rsp+10h] [rbp-40h]
  size_t n; // [rsp+10h] [rbp-40h]
  size_t nb; // [rsp+10h] [rbp-40h]

  v7 = a3 - 16;
  result = *(unsigned __int8 *)(a3 - 16);
  if ( (result & 2) != 0 )
  {
    v9 = *(_QWORD *)(a3 - 32);
  }
  else
  {
    result = 8LL * (((unsigned __int8)result >> 2) & 0xF);
    v9 = v7 - result;
  }
  v10 = *(_QWORD *)(v9 + 16);
  if ( v10 )
  {
    result = sub_B91420(v10);
    if ( v11 )
    {
      result = *(unsigned int *)(a3 + 20);
      if ( (result & 4) == 0 )
      {
        v12 = *(_BYTE *)(a3 - 16);
        v13 = *(_QWORD *)(a1 + 208);
        if ( (v12 & 2) != 0 )
          v14 = *(_QWORD *)(a3 - 32);
        else
          v14 = v7 - 8LL * ((v12 >> 2) & 0xF);
        v15 = *(_QWORD *)(v14 + 16);
        if ( v15 )
        {
          na = *(_QWORD *)(a1 + 208);
          v16 = sub_B91420(*(_QWORD *)(v14 + 16));
          v13 = na;
          v15 = v16;
          v18 = v17;
        }
        else
        {
          v18 = 0;
        }
        sub_3238440(v13, a1, *(_DWORD *)(*(_QWORD *)(a1 + 80) + 36LL), v15, v18, a4);
        if ( *(_BYTE *)a3 != 14 )
          return sub_3248250(a1, a3, a4, a2);
        v19 = *(_BYTE *)(a3 - 16);
        if ( (v19 & 2) != 0 )
        {
          v27 = *(_QWORD *)(a3 - 32);
          v20 = *(_QWORD *)(v27 + 56);
          if ( !v20 )
          {
            v24 = *(_QWORD *)(v27 + 16);
            if ( v24 )
            {
              v22 = 0;
              goto LABEL_16;
            }
            return sub_3248250(a1, a3, a4, a2);
          }
        }
        else
        {
          v20 = *(_QWORD *)(v7 - 8LL * ((v19 >> 2) & 0xF) + 56);
          if ( !v20 )
          {
            v22 = 0;
            goto LABEL_34;
          }
        }
        v20 = sub_B91420(v20);
        v19 = *(_BYTE *)(a3 - 16);
        v22 = v21;
        if ( (v19 & 2) != 0 )
        {
          v23 = *(_QWORD *)(a3 - 32);
LABEL_15:
          v24 = *(_QWORD *)(v23 + 16);
          if ( v24 )
          {
LABEL_16:
            s2 = (void *)v20;
            n = v22;
            v25 = (const void *)sub_B91420(v24);
            if ( n == v26 && (!n || !memcmp(v25, s2, n)) )
              return sub_3248250(a1, a3, a4, a2);
LABEL_27:
            if ( *(_DWORD *)(a3 + 44) == 30 )
            {
              v28 = *(_BYTE *)(a3 - 16);
              v29 = *(_QWORD *)(a1 + 208);
              if ( (v28 & 2) != 0 )
                v30 = *(_QWORD *)(a3 - 32);
              else
                v30 = v7 - 8LL * ((v28 >> 2) & 0xF);
              v31 = *(_QWORD *)(v30 + 56);
              if ( v31 )
              {
                nb = *(_QWORD *)(a1 + 208);
                v32 = sub_B91420(*(_QWORD *)(v30 + 56));
                v29 = nb;
                v31 = v32;
                v34 = v33;
              }
              else
              {
                v34 = 0;
              }
              sub_3238440(v29, a1, *(_DWORD *)(*(_QWORD *)(a1 + 80) + 36LL), v31, v34, a4);
            }
            return sub_3248250(a1, a3, a4, a2);
          }
          if ( v22 )
            goto LABEL_27;
          return sub_3248250(a1, a3, a4, a2);
        }
LABEL_34:
        v23 = v7 - 8LL * ((v19 >> 2) & 0xF);
        goto LABEL_15;
      }
    }
  }
  return result;
}
