// Function: sub_2DCFEE0
// Address: 0x2dcfee0
//
__int64 __fastcall sub_2DCFEE0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v5; // rbx
  __int64 v6; // r15
  __int64 v7; // rax
  _BYTE *v8; // r9
  size_t v9; // r8
  _QWORD *v10; // rax
  int v11; // eax
  int v12; // eax
  int v13; // eax
  __int64 v14; // rsi
  unsigned int v15; // r8d
  _QWORD *v16; // r9
  __int64 v17; // rax
  _QWORD *v18; // rdx
  _QWORD *v19; // r8
  __int64 (__fastcall *v20)(_QWORD *); // rax
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // r8
  __int64 (__fastcall *v23)(_QWORD *); // rax
  unsigned __int64 v24; // rdi
  _BYTE *v26; // rax
  _QWORD *v27; // rdi
  __int64 v28; // rax
  size_t v29; // rdx
  unsigned int v30; // r8d
  _QWORD *v31; // r9
  _QWORD *v32; // rcx
  __int64 *v33; // rdx
  _QWORD *v34; // [rsp+8h] [rbp-88h]
  _QWORD *v35; // [rsp+10h] [rbp-80h]
  _BYTE *v36; // [rsp+18h] [rbp-78h]
  unsigned int v37; // [rsp+18h] [rbp-78h]
  size_t na; // [rsp+20h] [rbp-70h]
  size_t n; // [rsp+20h] [rbp-70h]
  size_t nb; // [rsp+20h] [rbp-70h]
  void *src; // [rsp+28h] [rbp-68h]
  _BYTE *srca; // [rsp+28h] [rbp-68h]
  _QWORD *srcb; // [rsp+28h] [rbp-68h]
  void *srcc; // [rsp+28h] [rbp-68h]
  _BYTE *srcd; // [rsp+28h] [rbp-68h]
  _QWORD *v46; // [rsp+38h] [rbp-58h] BYREF
  _BYTE *v47; // [rsp+40h] [rbp-50h] BYREF
  size_t v48; // [rsp+48h] [rbp-48h]
  _QWORD v49[8]; // [rsp+50h] [rbp-40h] BYREF

  v3 = a3 + 24;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0x1000000000LL;
  v5 = *(_QWORD *)(a3 + 32);
  if ( v5 != a3 + 24 )
  {
    do
    {
      v6 = v5 - 56;
      if ( !v5 )
        v6 = 0;
      if ( sub_B2FC80(v6) || (*(_BYTE *)(v6 + 3) & 0x40) == 0 )
        goto LABEL_6;
      v7 = sub_B2DBE0(v6);
      v47 = v49;
      v8 = *(_BYTE **)v7;
      v9 = *(_QWORD *)(v7 + 8);
      if ( v9 + *(_QWORD *)v7 && !v8 )
        sub_426248((__int64)"basic_string::_M_construct null not valid");
      v46 = *(_QWORD **)(v7 + 8);
      if ( v9 > 0xF )
      {
        nb = v9;
        srcd = v8;
        v26 = (_BYTE *)sub_22409D0((__int64)&v47, (unsigned __int64 *)&v46, 0);
        v8 = srcd;
        v9 = nb;
        v47 = v26;
        v27 = v26;
        v49[0] = v46;
      }
      else
      {
        if ( v9 == 1 )
        {
          LOBYTE(v49[0]) = *v8;
          v10 = v49;
          goto LABEL_16;
        }
        if ( !v9 )
        {
          v10 = v49;
          goto LABEL_16;
        }
        v27 = v49;
      }
      memcpy(v27, v8, v9);
      v9 = (size_t)v46;
      v10 = v47;
LABEL_16:
      v48 = v9;
      *((_BYTE *)v10 + v9) = 0;
      v36 = v47;
      na = v48;
      src = (void *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8));
      v11 = sub_C92610();
      v12 = sub_C92860((__int64 *)a1, v36, na, v11);
      if ( v12 == -1 )
      {
        if ( src == (void *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8)) )
        {
LABEL_18:
          sub_E3FC80((__int64)&v46, v47, v48);
          n = v48;
          srca = v47;
          v13 = sub_C92610();
          v14 = (__int64)srca;
          v15 = sub_C92740(a1, srca, n, v13);
          v16 = (_QWORD *)(*(_QWORD *)a1 + 8LL * v15);
          v17 = *v16;
          if ( *v16 )
          {
            if ( v17 != -8 )
            {
LABEL_20:
              v18 = v46;
              v46 = 0;
              v19 = *(_QWORD **)(v17 + 8);
              *(_QWORD *)(v17 + 8) = v18;
              if ( v19 )
              {
                v20 = *(__int64 (__fastcall **)(_QWORD *))(*v19 + 8LL);
                if ( v20 == sub_BD9990 )
                {
                  v21 = v19[1];
                  *v19 = &unk_49DB390;
                  if ( (_QWORD *)v21 != v19 + 3 )
                  {
                    srcb = v19;
                    j_j___libc_free_0(v21);
                    v19 = srcb;
                  }
                  v14 = 48;
                  j_j___libc_free_0((unsigned __int64)v19);
                }
                else
                {
                  v20(v19);
                }
                v22 = (unsigned __int64)v46;
                if ( v46 )
                {
                  v23 = *(__int64 (__fastcall **)(_QWORD *))(*v46 + 8LL);
                  if ( v23 == sub_BD9990 )
                  {
                    v24 = v46[1];
                    *v46 = &unk_49DB390;
                    if ( v24 != v22 + 24 )
                    {
                      srcc = (void *)v22;
                      j_j___libc_free_0(v24);
                      v22 = (unsigned __int64)srcc;
                    }
                    j_j___libc_free_0(v22);
                  }
                  else
                  {
                    ((void (__fastcall *)(_QWORD *, __int64))v23)(v46, v14);
                  }
                }
              }
              goto LABEL_4;
            }
            --*(_DWORD *)(a1 + 16);
          }
          v35 = v16;
          v37 = v15;
          v28 = sub_C7D670(n + 17, 8);
          v29 = n;
          v30 = v37;
          v31 = v35;
          v32 = (_QWORD *)v28;
          if ( n )
          {
            v34 = (_QWORD *)v28;
            memcpy((void *)(v28 + 16), srca, n);
            v29 = n;
            v30 = v37;
            v31 = v35;
            v32 = v34;
          }
          *((_BYTE *)v32 + v29 + 16) = 0;
          v14 = v30;
          *v32 = v29;
          v32[1] = 0;
          *v31 = v32;
          ++*(_DWORD *)(a1 + 12);
          v33 = (__int64 *)(*(_QWORD *)a1 + 8LL * (unsigned int)sub_C929D0((__int64 *)a1, v30));
          v17 = *v33;
          if ( !*v33 || v17 == -8 )
          {
            do
            {
              do
              {
                v17 = v33[1];
                ++v33;
              }
              while ( v17 == -8 );
            }
            while ( !v17 );
          }
          goto LABEL_20;
        }
      }
      else if ( src == (void *)(*(_QWORD *)a1 + 8LL * v12) )
      {
        goto LABEL_18;
      }
LABEL_4:
      if ( v47 != (_BYTE *)v49 )
        j_j___libc_free_0((unsigned __int64)v47);
LABEL_6:
      v5 = *(_QWORD *)(v5 + 8);
    }
    while ( v3 != v5 );
  }
  return a1;
}
