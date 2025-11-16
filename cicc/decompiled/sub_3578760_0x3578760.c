// Function: sub_3578760
// Address: 0x3578760
//
void __fastcall sub_3578760(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  _BYTE *v6; // rbx
  _BYTE *v7; // r15
  _BYTE *v8; // r12
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 (*v11)(); // rax
  __int64 v12; // rbx
  __int64 v13; // r15
  __int64 v14; // rdx
  __int64 v15; // rax
  _QWORD *v16; // rdi
  _QWORD *v17; // rsi
  __int64 v18; // rcx
  _BYTE *v19; // rbx
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rdi
  __int64 v23; // rdx
  int v24; // eax
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // rdi
  __int64 v28; // rax
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // rdi
  __int64 v31; // rdx
  int v32; // r15d
  unsigned int v33; // r10d
  __int64 v34; // [rsp+0h] [rbp-70h]
  __int64 v35; // [rsp+8h] [rbp-68h]
  _QWORD *v36; // [rsp+10h] [rbp-60h]
  __int64 v37; // [rsp+18h] [rbp-58h]
  _BYTE *v38; // [rsp+20h] [rbp-50h]
  int v39; // [rsp+2Ch] [rbp-44h]
  __int64 v40[7]; // [rsp+38h] [rbp-38h] BYREF

  v36 = *(_QWORD **)(*(_QWORD *)(a1 + 216) + 32LL);
  v5 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*v36 + 16LL) + 200LL))(*(_QWORD *)(*v36 + 16LL));
  v6 = *(_BYTE **)(a2 + 32);
  v37 = v5;
  v7 = &v6[40 * (*(_DWORD *)(a2 + 40) & 0xFFFFFF)];
  if ( v6 != v7 )
  {
    while ( 1 )
    {
      v8 = v6;
      if ( sub_2DADC00(v6) )
        break;
      v6 += 40;
      if ( v7 == v6 )
        return;
    }
    if ( v7 != v6 )
    {
      v38 = v7;
      v35 = a1 + 16;
      while ( 1 )
      {
        v39 = *((_DWORD *)v8 + 2);
        if ( v39 < 0 )
          goto LABEL_49;
        v11 = *(__int64 (**)())(*(_QWORD *)v37 + 168LL);
        if ( v11 == sub_2EA3FB0 )
          goto LABEL_9;
        if ( !((unsigned __int8 (__fastcall *)(__int64, _QWORD))v11)(v37, (unsigned int)v39) )
          break;
LABEL_17:
        v19 = v8 + 40;
        if ( v8 + 40 != v38 )
        {
          while ( 1 )
          {
            v8 = v19;
            if ( sub_2DADC00(v19) )
              break;
            v19 += 40;
            if ( v38 == v19 )
              return;
          }
          if ( v38 != v19 )
            continue;
        }
        return;
      }
      v39 = *((_DWORD *)v8 + 2);
      if ( v39 >= 0 )
LABEL_9:
        v12 = *(_QWORD *)(v36[38] + 8LL * (unsigned int)v39);
      else
LABEL_49:
        v12 = *(_QWORD *)(v36[7] + 16LL * (v39 & 0x7FFFFFFF) + 8);
      if ( v12 )
      {
        if ( (*(_BYTE *)(v12 + 3) & 0x10) == 0 )
        {
LABEL_12:
          v13 = *(_QWORD *)(v12 + 16);
          v14 = *(unsigned int *)(a3 + 72);
          v15 = *(_QWORD *)(v13 + 24);
          v40[0] = v15;
          if ( (_DWORD)v14 )
          {
            while ( 1 )
            {
              v21 = *(_QWORD *)(a3 + 64);
              v18 = *(unsigned int *)(a3 + 80);
              v22 = v21 + 8 * v18;
              if ( !(_DWORD)v18 )
                goto LABEL_31;
              v18 = (unsigned int)(v18 - 1);
              v14 = (unsigned int)v18 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
              v9 = v21 + 8 * v14;
              v10 = *(_QWORD *)v9;
              if ( v15 != *(_QWORD *)v9 )
                break;
LABEL_30:
              if ( v22 == v9 )
                goto LABEL_31;
              while ( 1 )
              {
LABEL_16:
                v12 = *(_QWORD *)(v12 + 32);
                if ( !v12 )
                  goto LABEL_17;
                if ( (*(_BYTE *)(v12 + 3) & 0x10) == 0 )
                {
                  v20 = *(_QWORD *)(v12 + 16);
                  if ( v13 != v20 )
                    break;
                }
              }
              v13 = *(_QWORD *)(v12 + 16);
              v14 = *(unsigned int *)(a3 + 72);
              v15 = *(_QWORD *)(v20 + 24);
              v40[0] = v15;
              if ( !(_DWORD)v14 )
                goto LABEL_13;
            }
            v9 = 1;
            while ( v10 != -4096 )
            {
              v33 = v9 + 1;
              v14 = (unsigned int)v18 & ((_DWORD)v14 + (_DWORD)v9);
              v9 = v21 + 8LL * (unsigned int)v14;
              v10 = *(_QWORD *)v9;
              if ( v15 == *(_QWORD *)v9 )
                goto LABEL_30;
              v9 = v33;
            }
          }
          else
          {
LABEL_13:
            v16 = *(_QWORD **)(a3 + 88);
            v17 = &v16[*(unsigned int *)(a3 + 96)];
            if ( v17 != sub_3574250(v16, (__int64)v17, v40) )
              goto LABEL_16;
          }
LABEL_31:
          sub_3577FF0(a1, v13, v14, v18, v9, v10);
          v23 = *(unsigned int *)(a1 + 8);
          v24 = v23;
          if ( *(_DWORD *)(a1 + 12) <= (unsigned int)v23 )
          {
            v26 = sub_C8D7D0(a1, v35, 0, 0x18u, (unsigned __int64 *)v40, v10);
            v27 = 24LL * *(unsigned int *)(a1 + 8);
            v28 = v27 + v26;
            if ( v27 + v26 )
            {
              *(_QWORD *)v28 = a3;
              *(_QWORD *)(v28 + 8) = v13;
              *(_DWORD *)(v28 + 16) = v39;
              v27 = 24LL * *(unsigned int *)(a1 + 8);
            }
            v29 = *(_QWORD *)a1;
            v30 = *(_QWORD *)a1 + v27;
            if ( *(_QWORD *)a1 != v30 )
            {
              v31 = v26;
              do
              {
                if ( v31 )
                {
                  *(_QWORD *)v31 = *(_QWORD *)v29;
                  *(_QWORD *)(v31 + 8) = *(_QWORD *)(v29 + 8);
                  *(_DWORD *)(v31 + 16) = *(_DWORD *)(v29 + 16);
                }
                v29 += 24LL;
                v31 += 24;
              }
              while ( v30 != v29 );
              v30 = *(_QWORD *)a1;
            }
            v32 = v40[0];
            if ( v30 != v35 )
            {
              v34 = v26;
              _libc_free(v30);
              v26 = v34;
            }
            ++*(_DWORD *)(a1 + 8);
            *(_QWORD *)a1 = v26;
            *(_DWORD *)(a1 + 12) = v32;
          }
          else
          {
            v25 = *(_QWORD *)a1 + 24 * v23;
            if ( v25 )
            {
              *(_QWORD *)v25 = a3;
              *(_QWORD *)(v25 + 8) = v13;
              *(_DWORD *)(v25 + 16) = v39;
              v24 = *(_DWORD *)(a1 + 8);
            }
            *(_DWORD *)(a1 + 8) = v24 + 1;
          }
          v13 = *(_QWORD *)(v12 + 16);
          goto LABEL_16;
        }
        while ( 1 )
        {
          v12 = *(_QWORD *)(v12 + 32);
          if ( !v12 )
            break;
          if ( (*(_BYTE *)(v12 + 3) & 0x10) == 0 )
            goto LABEL_12;
        }
      }
      goto LABEL_17;
    }
  }
}
