// Function: sub_7B06F0
// Address: 0x7b06f0
//
char *__fastcall sub_7B06F0(char *a1, int a2, int a3, int a4)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  int v6; // ebx
  char *result; // rax
  int v8; // [rsp+8h] [rbp-38h] BYREF
  int v9; // [rsp+Ch] [rbp-34h] BYREF
  char *v10; // [rsp+10h] [rbp-30h] BYREF
  FILE *stream; // [rsp+18h] [rbp-28h] BYREF
  __int64 v12; // [rsp+20h] [rbp-20h] BYREF
  int v13[4]; // [rsp+28h] [rbp-18h] BYREF

  if ( a3 && (v4 = *(_QWORD *)(unk_4F064B0 + 32LL)) != 0 )
  {
    v5 = *(_QWORD *)(v4 + 16);
  }
  else if ( a2 )
  {
    v5 = unk_4F07688;
  }
  else
  {
    v5 = qword_4F076A8;
  }
  v6 = sub_7AFFB0(a1, 1, v5, (__int64 **)qword_4F084F0, 0, 0, &v10, (__int64 *)&stream, &v8, v13, &v9, &v12, a4);
  if ( stream )
    fclose(stream);
  result = 0;
  if ( v6 )
    return v10;
  return result;
}
