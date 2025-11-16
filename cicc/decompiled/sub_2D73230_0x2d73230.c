// Function: sub_2D73230
// Address: 0x2d73230
//
__int64 __fastcall sub_2D73230(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 result; // rax
  __int64 v5; // rcx
  __int64 v6; // rdx
  __int64 v7; // rsi
  unsigned int v8; // ecx
  _QWORD *v9; // rbx
  __int64 v10; // rdi
  __int64 v11; // r12
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rax
  bool v15; // zf
  __int64 v16; // r12
  __int64 v17; // rsi
  int v18; // r9d
  unsigned int v19; // esi
  int v20; // eax
  int v21; // eax
  __int64 v22; // rax
  __int64 v23; // [rsp+10h] [rbp-160h]
  __int64 v24; // [rsp+10h] [rbp-160h]
  __int64 v25; // [rsp+20h] [rbp-150h] BYREF
  __int64 v26; // [rsp+28h] [rbp-148h] BYREF
  unsigned __int64 v27[4]; // [rsp+30h] [rbp-140h] BYREF
  __int64 v28; // [rsp+50h] [rbp-120h]
  unsigned __int64 v29[3]; // [rsp+58h] [rbp-118h] BYREF
  void *v30; // [rsp+70h] [rbp-100h]
  unsigned __int64 v31[2]; // [rsp+78h] [rbp-F8h] BYREF
  __int64 v32; // [rsp+88h] [rbp-E8h]
  __int64 v33; // [rsp+90h] [rbp-E0h]
  void *v34; // [rsp+A0h] [rbp-D0h]
  _QWORD v35[3]; // [rsp+A8h] [rbp-C8h] BYREF
  __int64 v36; // [rsp+C0h] [rbp-B0h]
  void *v37; // [rsp+D0h] [rbp-A0h]
  _QWORD v38[2]; // [rsp+D8h] [rbp-98h] BYREF
  __int64 v39; // [rsp+E8h] [rbp-88h]
  __int64 v40; // [rsp+F0h] [rbp-80h]
  void *v41; // [rsp+100h] [rbp-70h] BYREF
  unsigned __int64 v42[2]; // [rsp+108h] [rbp-68h] BYREF
  __int64 v43; // [rsp+118h] [rbp-58h]
  __int64 v44; // [rsp+120h] [rbp-50h]
  unsigned __int64 v45[9]; // [rsp+128h] [rbp-48h] BYREF

  v3 = a1[1];
  v31[1] = 0;
  v31[0] = v3 & 6;
  v32 = a1[3];
  result = v32;
  if ( v32 != -4096 && v32 != 0 && v32 != -8192 )
  {
    sub_BD6050(v31, v3 & 0xFFFFFFFFFFFFFFF8LL);
    result = v32;
  }
  v5 = a1[4];
  v33 = v5;
  v30 = &unk_4A26638;
  v6 = *(unsigned int *)(v5 + 24);
  if ( (_DWORD)v6 )
  {
    v7 = *(_QWORD *)(v5 + 8);
    v8 = (v6 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
    v9 = (_QWORD *)(v7 + ((unsigned __int64)v8 << 6));
    v10 = v9[3];
    if ( result == v10 )
    {
LABEL_6:
      if ( v9 == (_QWORD *)(v7 + (v6 << 6)) )
        goto LABEL_19;
      sub_D68CD0(v27, 3u, v9 + 5);
      v11 = v33;
      sub_D68D70(v9 + 5);
      v12 = 0;
      v42[0] = 2;
      v42[1] = 0;
      v43 = -8192;
      v41 = &unk_4A26638;
      v44 = 0;
      v13 = v9[3];
      if ( v13 != -8192 )
      {
        if ( !v13 || v13 == -4096 )
        {
          v9[3] = -8192;
          v12 = v44;
        }
        else
        {
          sub_BD60C0(v9 + 1);
          v14 = v43;
          v15 = v43 == 0;
          v9[3] = v43;
          if ( v14 != -4096 && !v15 && v14 != -8192 )
            sub_BD6050(v9 + 1, v42[0] & 0xFFFFFFFFFFFFFFF8LL);
          v12 = v44;
        }
      }
      v9[4] = v12;
      v41 = &unk_49DB368;
      sub_D68D70(v42);
      --*(_DWORD *)(v11 + 16);
      ++*(_DWORD *)(v11 + 20);
      v16 = v33;
      v28 = a2;
      sub_D68CD0(v29, 3u, v27);
      v35[0] = 2;
      v35[1] = 0;
      v35[2] = v28;
      if ( v28 == 0 || v28 == -4096 || v28 == -8192 )
      {
        LODWORD(v17) = 1;
      }
      else
      {
        sub_BD73F0((__int64)v35);
        v17 = (v35[0] >> 1) & 3LL;
      }
      v36 = v16;
      v34 = &unk_4A26638;
      sub_D68CD0(v42, v17, v35);
      v41 = &unk_4A26638;
      v44 = v36;
      sub_D68CD0(v45, 3u, v29);
      if ( (unsigned __int8)sub_2D69820(v16, (__int64)&v41, &v25) )
        goto LABEL_25;
      v19 = *(_DWORD *)(v16 + 24);
      v26 = v25;
      v20 = *(_DWORD *)(v16 + 16);
      ++*(_QWORD *)v16;
      v21 = v20 + 1;
      if ( 4 * v21 >= 3 * v19 )
      {
        v19 *= 2;
      }
      else if ( v19 - *(_DWORD *)(v16 + 20) - v21 > v19 >> 3 )
      {
        goto LABEL_29;
      }
      sub_2D72DF0(v16, v19);
      sub_2D69820(v16, (__int64)&v41, &v26);
      v21 = *(_DWORD *)(v16 + 16) + 1;
LABEL_29:
      *(_DWORD *)(v16 + 16) = v21;
      v22 = v26;
      v39 = -4096;
      v40 = 0;
      v15 = *(_QWORD *)(v26 + 24) == -4096;
      v38[0] = 2;
      v38[1] = 0;
      if ( !v15 )
      {
        --*(_DWORD *)(v16 + 20);
        v37 = &unk_49DB368;
        if ( v39 != -8192 && v39 != -4096 )
        {
          if ( v39 )
          {
            v23 = v22;
            sub_BD60C0(v38);
            v22 = v23;
          }
        }
      }
      v24 = v22;
      sub_2D57190((unsigned __int64 *)(v22 + 8), v42);
      *(_QWORD *)(v24 + 32) = v44;
      sub_D68CD0((unsigned __int64 *)(v24 + 40), 3u, v45);
LABEL_25:
      sub_D68D70(v45);
      v41 = &unk_49DB368;
      sub_D68D70(v42);
      v34 = &unk_49DB368;
      sub_D68D70(v35);
      sub_D68D70(v29);
      sub_D68D70(v27);
      result = v32;
      goto LABEL_19;
    }
    v18 = 1;
    while ( v10 != -4096 )
    {
      v8 = (v6 - 1) & (v18 + v8);
      v9 = (_QWORD *)(v7 + ((unsigned __int64)v8 << 6));
      v10 = v9[3];
      if ( v10 == result )
        goto LABEL_6;
      ++v18;
    }
  }
LABEL_19:
  v30 = &unk_49DB368;
  if ( result != 0 && result != -4096 && result != -8192 )
    return sub_BD60C0(v31);
  return result;
}
