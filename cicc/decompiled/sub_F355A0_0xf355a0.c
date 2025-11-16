// Function: sub_F355A0
// Address: 0xf355a0
//
__int64 __fastcall sub_F355A0(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  __int64 v6; // r12
  __int64 v7; // r8
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // r15
  __int64 v12; // r12
  char *v13; // rdi
  char v14; // al
  _BYTE *v15; // r10
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rsi
  int v24; // edx
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rsi
  __int64 v28; // rdi
  __int64 v29; // rsi
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rsi
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // [rsp+0h] [rbp-80h]
  _QWORD *v40; // [rsp+8h] [rbp-78h]
  __int64 v41; // [rsp+10h] [rbp-70h]
  _QWORD *v42; // [rsp+18h] [rbp-68h]
  _BYTE *v44; // [rsp+28h] [rbp-58h]
  __int64 v45; // [rsp+28h] [rbp-58h]
  __int64 v46; // [rsp+30h] [rbp-50h]
  _QWORD v48[8]; // [rsp+40h] [rbp-40h] BYREF

  v42 = (_QWORD *)sub_986580(a3);
  v6 = sub_B47F80(a1);
  sub_B44240((_QWORD *)v6, a3, (unsigned __int64 *)(a3 + 48), 0);
  if ( (*(_BYTE *)(v6 + 7) & 0x40) != 0 )
  {
    v7 = *(_QWORD *)(v6 - 8);
    v8 = v7 + 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF);
  }
  else
  {
    v8 = v6;
    v7 = v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF);
  }
  if ( v7 != v8 )
  {
    v9 = v4;
    v46 = v6;
    v10 = v7;
    v11 = v9;
    while ( 1 )
    {
      v13 = *(char **)v10;
      v14 = **(_BYTE **)v10;
      if ( v14 == 78 )
      {
        LOWORD(v11) = 0;
        v44 = (_BYTE *)*((_QWORD *)v13 - 4);
        v12 = sub_B47F80(v13);
        sub_B44240((_QWORD *)v12, a3, (unsigned __int64 *)(v46 + 24), v11);
        v15 = v44;
        if ( *(_QWORD *)v10 )
        {
          v16 = *(_QWORD *)(v10 + 8);
          **(_QWORD **)(v10 + 16) = v16;
          if ( v16 )
            *(_QWORD *)(v16 + 16) = *(_QWORD *)(v10 + 16);
        }
        *(_QWORD *)v10 = v12;
        if ( v12 )
        {
          v17 = *(_QWORD *)(v12 + 16);
          *(_QWORD *)(v10 + 8) = v17;
          if ( v17 )
            *(_QWORD *)(v17 + 16) = v10 + 8;
          *(_QWORD *)(v10 + 16) = v12 + 16;
          *(_QWORD *)(v12 + 16) = v10;
          v14 = *v44;
          if ( *v44 == 93 )
          {
            v45 = *((_QWORD *)v44 - 4);
            v18 = sub_B47F80(v15);
            if ( (*(_BYTE *)(v12 + 7) & 0x40) != 0 )
              v19 = *(_QWORD *)(v12 - 8);
            else
              v19 = v12 - 32LL * (*(_DWORD *)(v12 + 4) & 0x7FFFFFF);
            if ( *(_QWORD *)v19 )
            {
              v20 = *(_QWORD *)(v19 + 8);
              **(_QWORD **)(v19 + 16) = v20;
              if ( v20 )
                *(_QWORD *)(v20 + 16) = *(_QWORD *)(v19 + 16);
            }
            *(_QWORD *)v19 = v18;
            if ( v18 )
            {
              v21 = *(_QWORD *)(v18 + 16);
              *(_QWORD *)(v19 + 8) = v21;
              if ( v21 )
                *(_QWORD *)(v21 + 16) = v19 + 8;
              *(_QWORD *)(v19 + 16) = v18 + 16;
              *(_QWORD *)(v18 + 16) = v19;
            }
            v22 = v41;
            LOWORD(v22) = 0;
            v41 = v22;
            sub_B44240((_QWORD *)v18, a3, (unsigned __int64 *)(v12 + 24), v22);
            goto LABEL_27;
          }
          v13 = v44;
          goto LABEL_7;
        }
        v14 = *v44;
        v13 = v44;
      }
      if ( v14 == 93 )
      {
        v45 = *((_QWORD *)v13 - 4);
        v40 = (_QWORD *)sub_B47F80(v13);
        v33 = v39;
        LOWORD(v33) = 0;
        v39 = v33;
        sub_B44240(v40, a3, (unsigned __int64 *)(v46 + 24), v33);
        v18 = (__int64)v40;
        if ( *(_QWORD *)v10 )
        {
          v34 = *(_QWORD *)(v10 + 8);
          **(_QWORD **)(v10 + 16) = v34;
          if ( v34 )
            *(_QWORD *)(v34 + 16) = *(_QWORD *)(v10 + 16);
        }
        *(_QWORD *)v10 = v40;
        if ( !v40 )
        {
          if ( *(_BYTE *)v45 == 84 && a2 == *(_QWORD *)(v45 + 40) )
          {
            v13 = (char *)v45;
LABEL_64:
            v36 = *((_QWORD *)v13 - 1);
            v37 = 0x1FFFFFFFE0LL;
            if ( (*((_DWORD *)v13 + 1) & 0x7FFFFFF) != 0 )
            {
              v38 = 0;
              do
              {
                if ( a3 == *(_QWORD *)(v36 + 32LL * *((unsigned int *)v13 + 18) + 8 * v38) )
                {
                  v37 = 32 * v38;
                  goto LABEL_69;
                }
                ++v38;
              }
              while ( (*((_DWORD *)v13 + 1) & 0x7FFFFFF) != (_DWORD)v38 );
              v37 = 0x1FFFFFFFE0LL;
            }
LABEL_69:
            v27 = *(_QWORD *)(v36 + v37);
            v28 = v10;
            goto LABEL_46;
          }
          goto LABEL_8;
        }
        v35 = v40[2];
        *(_QWORD *)(v10 + 8) = v35;
        if ( v35 )
          *(_QWORD *)(v35 + 16) = v10 + 8;
        *(_QWORD *)(v10 + 16) = v40 + 2;
        v12 = 0;
        v40[2] = v10;
LABEL_27:
        if ( *(_BYTE *)v45 != 84 || a2 != *(_QWORD *)(v45 + 40) )
          goto LABEL_8;
        v23 = *(_QWORD *)(v45 - 8);
        v24 = *(_DWORD *)(v45 + 4) & 0x7FFFFFF;
        if ( v18 )
        {
          v25 = 0x1FFFFFFFE0LL;
          if ( v24 )
          {
            v26 = 0;
            do
            {
              if ( a3 == *(_QWORD *)(v23 + 32LL * *(unsigned int *)(v45 + 72) + 8 * v26) )
              {
                v25 = 32 * v26;
                goto LABEL_35;
              }
              ++v26;
            }
            while ( v24 != (_DWORD)v26 );
            v25 = 0x1FFFFFFFE0LL;
          }
LABEL_35:
          v27 = *(_QWORD *)(v23 + v25);
          if ( (*(_BYTE *)(v18 + 7) & 0x40) != 0 )
            v28 = *(_QWORD *)(v18 - 8);
          else
            v28 = v18 - 32LL * (*(_DWORD *)(v18 + 4) & 0x7FFFFFF);
          goto LABEL_46;
        }
        v13 = (char *)v45;
LABEL_38:
        if ( !v12 )
          goto LABEL_64;
        v29 = *((_QWORD *)v13 - 1);
        v30 = 0x1FFFFFFFE0LL;
        if ( (*((_DWORD *)v13 + 1) & 0x7FFFFFF) != 0 )
        {
          v31 = 0;
          do
          {
            if ( a3 == *(_QWORD *)(v29 + 32LL * *((unsigned int *)v13 + 18) + 8 * v31) )
            {
              v30 = 32 * v31;
              goto LABEL_44;
            }
            ++v31;
          }
          while ( (*((_DWORD *)v13 + 1) & 0x7FFFFFF) != (_DWORD)v31 );
          v30 = 0x1FFFFFFFE0LL;
        }
LABEL_44:
        v27 = *(_QWORD *)(v29 + v30);
        if ( (*(_BYTE *)(v12 + 7) & 0x40) != 0 )
          v28 = *(_QWORD *)(v12 - 8);
        else
          v28 = v12 - 32LL * (*(_DWORD *)(v12 + 4) & 0x7FFFFFF);
LABEL_46:
        sub_AC2B30(v28, v27);
        v10 += 32;
        if ( v8 == v10 )
        {
LABEL_47:
          v6 = v46;
          break;
        }
      }
      else
      {
        v12 = 0;
LABEL_7:
        if ( v14 == 84 && a2 == *((_QWORD *)v13 + 5) )
          goto LABEL_38;
LABEL_8:
        v10 += 32;
        if ( v8 == v10 )
          goto LABEL_47;
      }
    }
  }
  sub_AA5980(a2, a3, 0);
  sub_B43D60(v42);
  if ( a4 )
  {
    v48[0] = a3;
    v48[1] = a2 | 4;
    sub_FFB3D0(a4, v48, 1);
  }
  return v6;
}
