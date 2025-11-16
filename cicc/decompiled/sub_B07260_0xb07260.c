// Function: sub_B07260
// Address: 0xb07260
//
__int64 __fastcall sub_B07260(__int64 *a1, int a2, char a3, __int64 a4, unsigned int a5, int a6)
{
  int v8; // r13d
  __int64 v10; // r10
  int v11; // ebx
  int v12; // eax
  int v13; // r8d
  int v14; // r11d
  unsigned int i; // esi
  __int64 v16; // rbx
  unsigned int v17; // esi
  __int64 result; // rax
  __int64 v19; // rbx
  __int64 v20; // rax
  __int64 v21; // rdi
  _BYTE *v22; // rax
  int v23; // [rsp+Ch] [rbp-84h]
  __int64 v24; // [rsp+10h] [rbp-80h]
  int v25; // [rsp+18h] [rbp-78h]
  int v26; // [rsp+1Ch] [rbp-74h]
  __int64 v27; // [rsp+20h] [rbp-70h]
  int v28; // [rsp+20h] [rbp-70h]
  __int64 v29; // [rsp+28h] [rbp-68h]
  __int64 v30; // [rsp+30h] [rbp-60h]
  __int64 v31; // [rsp+30h] [rbp-60h]
  __int128 v32; // [rsp+40h] [rbp-50h] BYREF
  __int64 v33; // [rsp+50h] [rbp-40h]
  __int64 v34; // [rsp+58h] [rbp-38h]

  v8 = (int)a1;
  if ( a5 )
    goto LABEL_9;
  v10 = *a1;
  LODWORD(v32) = a2;
  BYTE4(v32) = a3;
  *((_QWORD *)&v32 + 1) = a4;
  v11 = *(_DWORD *)(v10 + 1008);
  v29 = v10;
  v30 = *(_QWORD *)(v10 + 992);
  if ( v11 )
  {
    v26 = a6;
    v27 = a4;
    v12 = sub_AF8410((int *)&v32, (__int8 *)&v32 + 4, (__int64 *)&v32 + 1);
    v13 = v11 - 1;
    a4 = v27;
    a6 = v26;
    v14 = 1;
    for ( i = (v11 - 1) & v12; ; i = v13 & v17 )
    {
      v16 = *(_QWORD *)(v30 + 8LL * i);
      if ( v16 == -4096 )
        break;
      if ( v16 != -8192 && (_DWORD)v32 == *(_DWORD *)(v16 + 20) && BYTE4(v32) == *(_BYTE *)(v16 + 44) )
      {
        v23 = a6;
        v24 = a4;
        v25 = v14;
        v28 = v13;
        v22 = sub_A17150((_BYTE *)(v16 - 16));
        v13 = v28;
        v14 = v25;
        a4 = v24;
        a6 = v23;
        if ( *((_QWORD *)&v32 + 1) == *((_QWORD *)v22 + 3) )
        {
          if ( v30 + 8LL * i == *(_QWORD *)(v29 + 992) + 8LL * *(unsigned int *)(v29 + 1008) )
            break;
          return v16;
        }
      }
      v17 = v14 + i;
      ++v14;
    }
  }
  result = 0;
  if ( (_BYTE)a6 )
  {
LABEL_9:
    v19 = *a1 + 984;
    v33 = 0;
    v34 = a4;
    v32 = 0;
    v20 = sub_B97910(48, 4, a5);
    v21 = v20;
    if ( v20 )
    {
      v31 = v20;
      sub_AF2E80(v20, v8, a5, a2, a3, 4, (__int64)&v32, 4);
      v21 = v31;
    }
    return sub_B070C0(v21, a5, v19);
  }
  return result;
}
