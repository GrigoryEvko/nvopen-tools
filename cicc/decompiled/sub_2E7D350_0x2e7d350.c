// Function: sub_2E7D350
// Address: 0x2e7d350
//
__int64 __fastcall sub_2E7D350(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // r13
  char v14; // al
  __int64 v15; // r15
  __int64 v16; // rdi
  unsigned __int8 *v17; // rsi
  int v18; // eax
  bool v19; // cf
  unsigned __int8 **v20; // r8
  unsigned __int8 **v21; // r14
  unsigned __int8 **v22; // r13
  int v23; // eax
  __int64 v24; // r9
  __int64 v25; // rdx
  unsigned __int64 v26; // r8
  __int64 v27; // rdx
  _BYTE *v28; // rsi
  int v29; // eax
  __int64 v31; // rax
  __int64 v32; // r14
  unsigned __int8 *v33; // rsi
  int v34; // eax
  __int64 v35; // rax
  __int64 v36; // [rsp+8h] [rbp-98h]
  __int64 v37; // [rsp+10h] [rbp-90h]
  __int64 v38; // [rsp+20h] [rbp-80h]
  int v39; // [rsp+2Ch] [rbp-74h]
  __int64 v40; // [rsp+38h] [rbp-68h]
  int v41; // [rsp+4Ch] [rbp-54h] BYREF
  _BYTE *v42; // [rsp+50h] [rbp-50h] BYREF
  __int64 v43; // [rsp+58h] [rbp-48h]
  _BYTE v44[64]; // [rsp+60h] [rbp-40h] BYREF

  v37 = sub_E6C430(*(_QWORD *)(a1 + 24), (__int64)a2, a3, a4, a5);
  v10 = sub_2E7D0F0((unsigned __int64 *)a1, (__int64)a2, v6, v7, v8, v9);
  *(_QWORD *)(v10 + 88) = v37;
  v11 = v10;
  v12 = sub_AA4FF0(*((_QWORD *)a2 + 2));
  if ( !v12 )
    BUG();
  v13 = v12;
  v14 = *(_BYTE *)(v12 - 24);
  if ( v14 == 95 )
  {
    if ( (*(_BYTE *)(v13 - 22) & 1) != 0 )
    {
      if ( (*(_DWORD *)(v13 - 20) & 0x7FFFFFF) == 0 )
        return v37;
      LODWORD(v42) = 0;
      a2 = *(unsigned __int8 **)(v11 + 104);
      if ( a2 == *(unsigned __int8 **)(v11 + 112) )
      {
        sub_1E0CD40(v11 + 96, a2, &v42);
      }
      else
      {
        if ( a2 )
        {
          *(_DWORD *)a2 = 0;
          a2 = *(unsigned __int8 **)(v11 + 104);
        }
        a2 += 4;
        *(_QWORD *)(v11 + 104) = a2;
      }
    }
    if ( (*(_DWORD *)(v13 - 20) & 0x7FFFFFF) != 0 )
    {
      v15 = (*(_DWORD *)(v13 - 20) & 0x7FFFFFFu) - 1;
      v40 = v13;
      v36 = v11 + 96;
      v38 = v13 - 24;
      while ( 1 )
      {
        if ( (*(_BYTE *)(v40 - 17) & 0x40) != 0 )
        {
          v16 = *(_QWORD *)(*(_QWORD *)(v40 - 32) + 32 * v15);
          if ( *(_BYTE *)(*(_QWORD *)(v16 + 8) + 8LL) != 16 )
            goto LABEL_7;
        }
        else
        {
          v16 = *(_QWORD *)(v38 + 32 * (v15 - (*(_DWORD *)(v40 - 20) & 0x7FFFFFF)));
          if ( *(_BYTE *)(*(_QWORD *)(v16 + 8) + 8LL) != 16 )
          {
LABEL_7:
            v17 = sub_BD3990((unsigned __int8 *)v16, (__int64)a2);
            if ( *v17 >= 4u )
              v17 = 0;
            v18 = sub_2E7AA20((_QWORD *)a1, (__int64)v17);
            LODWORD(v42) = v18;
            a2 = *(unsigned __int8 **)(v11 + 104);
            if ( a2 == *(unsigned __int8 **)(v11 + 112) )
            {
              sub_1E0CD40(v36, a2, &v42);
            }
            else
            {
              if ( a2 )
              {
                *(_DWORD *)a2 = v18;
                a2 = *(unsigned __int8 **)(v11 + 104);
              }
              a2 += 4;
              *(_QWORD *)(v11 + 104) = a2;
            }
            goto LABEL_13;
          }
        }
        v42 = v44;
        v43 = 0x400000000LL;
        if ( (*(_BYTE *)(v16 + 7) & 0x40) != 0 )
        {
          v20 = *(unsigned __int8 ***)(v16 - 8);
          v21 = &v20[4 * (*(_DWORD *)(v16 + 4) & 0x7FFFFFF)];
          if ( v21 != v20 )
            goto LABEL_18;
        }
        else
        {
          v21 = (unsigned __int8 **)v16;
          v35 = 32LL * (*(_DWORD *)(v16 + 4) & 0x7FFFFFF);
          v20 = (unsigned __int8 **)(v16 - v35);
          if ( v16 != v16 - v35 )
          {
LABEL_18:
            v22 = v20;
            do
            {
              a2 = sub_BD3990(*v22, (__int64)a2);
              v23 = sub_2E7AA20((_QWORD *)a1, (__int64)a2);
              v25 = (unsigned int)v43;
              v26 = (unsigned int)v43 + 1LL;
              if ( v26 > HIDWORD(v43) )
              {
                a2 = v44;
                v39 = v23;
                sub_C8D5F0((__int64)&v42, v44, (unsigned int)v43 + 1LL, 4u, v26, v24);
                v25 = (unsigned int)v43;
                v23 = v39;
              }
              v22 += 4;
              *(_DWORD *)&v42[4 * v25] = v23;
              v27 = (unsigned int)(v43 + 1);
              LODWORD(v43) = v43 + 1;
            }
            while ( v21 != v22 );
            v28 = v42;
            goto LABEL_23;
          }
        }
        v28 = v44;
        v27 = 0;
LABEL_23:
        v29 = sub_2E7BEF0((_QWORD *)a1, v28, v27);
        a2 = *(unsigned __int8 **)(v11 + 104);
        v41 = v29;
        if ( a2 == *(unsigned __int8 **)(v11 + 112) )
        {
          sub_1E0CD40(v36, a2, &v41);
        }
        else
        {
          if ( a2 )
          {
            *(_DWORD *)a2 = v29;
            a2 = *(unsigned __int8 **)(v11 + 104);
          }
          a2 += 4;
          *(_QWORD *)(v11 + 104) = a2;
        }
        if ( v42 == v44 )
        {
LABEL_13:
          v19 = v15-- == 0;
          if ( v19 )
            return v37;
        }
        else
        {
          _libc_free((unsigned __int64)v42);
          v19 = v15-- == 0;
          if ( v19 )
            return v37;
        }
      }
    }
  }
  else if ( v14 == 81 )
  {
    v31 = *(_DWORD *)(v13 - 20) & 0x7FFFFFF;
    if ( (_DWORD)v31 != 1 )
    {
      v32 = (unsigned int)(v31 - 2);
      while ( 1 )
      {
        v33 = sub_BD3990(*(unsigned __int8 **)(v13 - 24 + 32 * (v32 - v31)), (__int64)a2);
        if ( *v33 >= 4u )
          v33 = 0;
        v34 = sub_2E7AA20((_QWORD *)a1, (__int64)v33);
        LODWORD(v42) = v34;
        a2 = *(unsigned __int8 **)(v11 + 104);
        if ( a2 == *(unsigned __int8 **)(v11 + 112) )
        {
          sub_1E0CD40(v11 + 96, a2, &v42);
        }
        else
        {
          if ( a2 )
          {
            *(_DWORD *)a2 = v34;
            a2 = *(unsigned __int8 **)(v11 + 104);
          }
          a2 += 4;
          *(_QWORD *)(v11 + 104) = a2;
        }
        v19 = v32-- == 0;
        if ( v19 )
          break;
        v31 = *(_DWORD *)(v13 - 20) & 0x7FFFFFF;
      }
    }
  }
  return v37;
}
