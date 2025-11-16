// Function: sub_2DC80E0
// Address: 0x2dc80e0
//
__int64 __fastcall sub_2DC80E0(_QWORD *a1, __int64 a2)
{
  __int64 (*v3)(void); // rdx
  __int64 v4; // rax
  unsigned int v5; // r13d
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // rdi
  __int64 (*v12)(); // rax
  unsigned __int16 v13; // ax
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // rdi
  __int64 i; // rdi
  __int64 v20; // rax
  _QWORD *v21; // rax
  __int64 v22; // r13
  _QWORD *v23; // rbx
  _QWORD *v24; // r15
  _QWORD *v25; // r12
  unsigned __int64 *v26; // rdi
  unsigned __int64 v27; // rdx
  unsigned int v29; // eax
  unsigned int v30; // [rsp+8h] [rbp-58h]
  unsigned int v31; // [rsp+Ch] [rbp-54h]
  unsigned int v32; // [rsp+10h] [rbp-50h]
  __int64 v33; // [rsp+10h] [rbp-50h]
  __int64 v34; // [rsp+18h] [rbp-48h]
  unsigned __int8 v35; // [rsp+18h] [rbp-48h]
  __int64 v36; // [rsp+20h] [rbp-40h]

  *a1 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16));
  v3 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 128LL);
  v4 = 0;
  if ( v3 != sub_2DAC790 )
    v4 = v3();
  a1[1] = v4;
  v5 = 0;
  v36 = *(_QWORD *)(a2 + 328);
  if ( v36 != a2 + 320 )
  {
    while ( 1 )
    {
      v6 = *(_QWORD *)(v36 + 56);
      if ( v6 != v36 + 48 )
        break;
LABEL_33:
      v36 = *(_QWORD *)(v36 + 8);
      if ( a2 + 320 == v36 )
        return v5;
    }
    while ( 1 )
    {
      if ( !v6 )
        BUG();
      v7 = v6;
      if ( (*(_BYTE *)v6 & 4) == 0 && (*(_BYTE *)(v6 + 44) & 8) != 0 )
      {
        do
          v7 = *(_QWORD *)(v7 + 8);
        while ( (*(_BYTE *)(v7 + 44) & 8) != 0 );
      }
      v8 = *(_QWORD *)(v7 + 8);
      v9 = *(_QWORD *)(v6 + 16);
      v10 = (*(_QWORD *)(v9 + 24) >> 3) & 1LL;
      if ( ((*(_QWORD *)(v9 + 24) >> 3) & 1) == 0 )
        goto LABEL_32;
      v11 = a1[1];
      v12 = *(__int64 (**)())(*(_QWORD *)v11 + 576LL);
      if ( v12 != sub_2DC7E00 )
      {
        v29 = ((__int64 (__fastcall *)(__int64, __int64))v12)(v11, v6);
        if ( (_BYTE)v29 )
        {
          v5 = v29;
          goto LABEL_32;
        }
      }
      v13 = *(_WORD *)(v6 + 68);
      if ( v13 == 20 )
        break;
      if ( v13 <= 0x14u )
      {
        if ( v13 <= 9u )
        {
          if ( v13 > 7u )
            BUG();
          goto LABEL_32;
        }
        if ( v13 == 12 )
        {
          v34 = *(_QWORD *)(v6 + 24);
          v14 = *(_QWORD *)(v6 + 32);
          v30 = *(_DWORD *)(v14 + 88);
          v31 = *(_DWORD *)(v14 + 8);
          v32 = sub_E91CF0((_QWORD *)*a1, v31, *(_QWORD *)(v14 + 144));
          v5 = sub_2E8B940(v6);
          if ( (_BYTE)v5 )
          {
            sub_2E88D70(v6, *(_QWORD *)(a1[1] + 8LL) - 280LL, v15, v16, v32, v30);
            sub_2E8A650(v6, 3);
            sub_2E8A650(v6, 1);
            goto LABEL_32;
          }
          if ( v30 == v32 )
          {
            if ( v31 != v30 )
            {
              v5 = v10;
              sub_2E88D70(v6, *(_QWORD *)(a1[1] + 8LL) - 280LL, v15, v16, v32, v30);
              sub_2E8A650(v6, 3);
              sub_2E8A650(v6, 1);
              goto LABEL_32;
            }
          }
          else
          {
            (*(void (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD))(*(_QWORD *)a1[1] + 496LL))(
              a1[1],
              v34,
              v6,
              v6 + 56,
              v32,
              v30,
              ((*(_BYTE *)(*(_QWORD *)(v6 + 32) + 83LL) >> 4) ^ 1) & (*(_BYTE *)(*(_QWORD *)(v6 + 32) + 83LL) >> 6) & 1,
              0,
              0);
            v17 = *(_QWORD *)v6 & 0xFFFFFFFFFFFFFFF8LL;
            if ( !v17 )
              BUG();
            v18 = *(_QWORD *)v6 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_QWORD *)v17 & 4) == 0 && (*(_BYTE *)(v17 + 44) & 4) != 0 )
            {
              for ( i = *(_QWORD *)v17; ; i = *(_QWORD *)v18 )
              {
                v18 = i & 0xFFFFFFFFFFFFFFF8LL;
                if ( (*(_BYTE *)(v18 + 44) & 4) == 0 )
                  break;
              }
            }
            sub_2E8FA40(v18, v31, 0);
          }
          v20 = v6;
          if ( (*(_BYTE *)v6 & 4) == 0 && (*(_BYTE *)(v6 + 44) & 8) != 0 )
          {
            do
              v20 = *(_QWORD *)(v20 + 8);
            while ( (*(_BYTE *)(v20 + 44) & 8) != 0 );
          }
          v21 = *(_QWORD **)(v20 + 8);
          v22 = v34 + 40;
          if ( (_QWORD *)v6 != v21 )
          {
            v33 = v8;
            v35 = v10;
            v23 = (_QWORD *)v6;
            v24 = v21;
            do
            {
              v25 = v23;
              v23 = (_QWORD *)v23[1];
              sub_2E31080(v22, v25);
              v26 = (unsigned __int64 *)v25[1];
              v27 = *v25 & 0xFFFFFFFFFFFFFFF8LL;
              *v26 = v27 | *v26 & 7;
              *(_QWORD *)(v27 + 8) = v26;
              *v25 &= 7uLL;
              v25[1] = 0;
              sub_2E310F0(v22, v25);
            }
            while ( v23 != v24 );
            LODWORD(v10) = v35;
            v8 = v33;
          }
          goto LABEL_31;
        }
      }
LABEL_32:
      v6 = v8;
      if ( v8 == v36 + 48 )
        goto LABEL_33;
    }
    sub_2FE00C0(a1[1], v6, *a1);
LABEL_31:
    v5 = v10;
    goto LABEL_32;
  }
  return v5;
}
